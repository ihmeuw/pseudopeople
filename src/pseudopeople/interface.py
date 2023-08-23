from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from tqdm import tqdm
from vivarium import ConfigTree

from pseudopeople.configuration import get_configuration
from pseudopeople.constants import paths
from pseudopeople.constants.metadata import COPY_HOUSEHOLD_MEMBER_COLS, DatasetNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.loader import load_standard_dataset
from pseudopeople.noise import noise_dataset
from pseudopeople.schema_entities import COLUMNS, DATASETS, Dataset
from pseudopeople.utilities import (
    PANDAS,
    DataFrame,
    configure_logging_to_terminal,
    get_engine_from_string,
    get_state_abbreviation,
)


def _generate_dataset(
    dataset: Dataset,
    source: Union[Path, str],
    seed: int,
    config: Union[Path, str, Dict],
    user_filters: List[tuple],
    verbose: bool = False,
    engine: Literal["pandas", "modin"] = "pandas",
) -> DataFrame:
    """
    Helper for generating noised datasets.

    :param dataset:
        Dataset needing to be noised
    :param source:
        Root directory of data input which needs to be noised
    :param seed:
        Seed for controlling randomness
    :param config:
        Object to configure noise levels
    :param user_filters:
        List of parquet filters, possibly empty
    :param verbose:
        Log with verbosity if True. Default is False.
    :param engine:
        Engine to use for loading data. Determines the return type.
    :return:
        Noised dataset data in a dataframe
    """
    configure_logging_to_terminal(verbose)
    configuration_tree = get_configuration(config)

    if source is None:
        source = paths.SAMPLE_DATA_ROOT
    else:
        source = Path(source)

    engine = get_engine_from_string(engine)

    if engine == PANDAS:
        # We process shards serially
        data_paths = fetch_filepaths(dataset, source)
        if not data_paths:
            raise DataSourceError(
                f"No datasets found at directory {str(source)}. "
                "Please provide the path to the unmodified root data directory."
            )

        validate_data_path_suffix(data_paths)

        # Iterate sequentially
        noised_dataset = []
        iterator = (
            tqdm(data_paths, desc="Noising data", leave=False)
            if len(data_paths) > 1
            else data_paths
        )

        for data_path in iterator:
            logger.debug(f"Loading data from {data_path}.")
            data = load_standard_dataset(data_path, user_filters, engine=engine, is_file=True)
            if len(data.index) == 0:
                continue
            noised_data = _prep_and_noise_dataset(data, dataset, configuration_tree, seed)
            noised_dataset.append(noised_data)

        noised_dataset = pd.concat(noised_dataset, ignore_index=True)

        # Known pandas bug: pd.concat does not preserve category dtypes so we coerce
        # again after concat (https://github.com/pandas-dev/pandas/issues/51362)
        noised_dataset = _coerce_dtypes(noised_dataset, dataset)
    else:
        import modin.pandas as mpd

        # Let modin deal with how to partition the shards -- the data path is the
        # entire directory containing the parquet files
        data_path = source / dataset.name
        data = load_standard_dataset(data_path, user_filters, engine=engine, is_file=False)

        # FIXME: This is using private Modin APIs
        #  How do we do this using the public API?
        #  See https://github.com/modin-project/modin/issues/6498
        from modin.core.storage_formats import PandasQueryCompiler

        noised_dataset = mpd.DataFrame(
            query_compiler=PandasQueryCompiler(
                data._query_compiler._modin_frame.apply_full_axis(
                    axis=1,
                    func=lambda df: _prep_and_noise_dataset(
                        df, dataset, configuration_tree, seed
                    ),
                )
            )
        )

    logger.debug("*** Finished ***")

    return noised_dataset


def _prep_and_noise_dataset(
    data: pd.DataFrame, dataset: Dataset, configuration_tree: ConfigTree, seed: int
) -> pd.DataFrame:
    data = _reformat_dates_for_noising(data, dataset)
    data = _coerce_dtypes(data, dataset)
    noised_data = noise_dataset(dataset, data, configuration_tree, seed)
    noised_data = _extract_columns(dataset.columns, noised_data)
    return noised_data


def _coerce_dtypes(data: pd.DataFrame, dataset: Dataset):
    # Coerce dtypes prior to noising to catch issues early as well as
    # get most columns away from dtype 'category' and into 'object' (strings)
    for col in dataset.columns:
        if col.dtype_name != data[col.name].dtype.name:
            data[col.name] = data[col.name].astype(col.dtype_name)
    return data


def _reformat_dates_for_noising(data: pd.DataFrame, dataset: Dataset):
    """Formats date columns so they can be noised as strings."""
    data = data.copy()

    for date_column in [COLUMNS.dob.name, COLUMNS.ssa_event_date.name]:
        # Format both the actual column, and the shadow version that will be used
        # to copy from a household member
        for column in [date_column, COPY_HOUSEHOLD_MEMBER_COLS.get(date_column)]:
            if column in data.columns:
                data[column] = data[column].dt.strftime(dataset.date_format)

    return data


def _extract_columns(columns_to_keep, noised_dataset):
    """Helper function for test mocking purposes"""
    if columns_to_keep:
        noised_dataset = noised_dataset[[c.name for c in columns_to_keep]]
    return noised_dataset


def generate_decennial_census(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "modin"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople decennial census dataset which represents simulated
    responses to the US Census Bureau's Census of Population and Housing.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The year (format YYYY) to include in the dataset. Must be a decennial
        year (e.g. 2020, 2030, 2040). Will return an empty DataFrame if there are no
        data with this year. If None is provided, data from all years are
        included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :param engine:
        Engine to use for loading data. Determines the return type.
    :return: A DataFrame of simulated decennial census data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append((DATASETS.census.date_column_name, "==", year))
    if state:
        user_filters.append(
            (DATASETS.census.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.census, source, seed, config, user_filters, verbose, engine=engine
    )


def generate_american_community_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "modin"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople ACS dataset which represents simulated responses to
    the ACS survey.

    The American Community Survey (ACS) is an ongoing household survey conducted by
    the US Census Bureau that gathers information on a rolling basis about
    American community populations. Information collected includes ancestry,
    citizenship, education, income, language proficiency, migration, employment,
    disability, and housing characteristics.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The survey date year (format YYYY) to include in the dataset. Will
        return an empty DataFrame if there are no data with this year. If None is
        provided, data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :param engine:
        Engine to use for loading data. Determines the return type.
    :return: A DataFrame of simulated ACS data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.extend(
            [
                (DATASETS.acs.date_column_name, ">=", pd.Timestamp(f"{year}-01-01")),
                (DATASETS.acs.date_column_name, "<=", pd.Timestamp(f"{year}-12-31")),
            ]
        )
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.acs.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.acs, source, seed, config, user_filters, verbose, engine=engine
    )


def generate_current_population_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "modin"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople CPS dataset which represents simulated responses to
    the CPS survey.

    The Current Population Survey (CPS) is a household survey conducted by the
    US Census Bureau and the US Bureau of Labor Statistics. This survey is administered
    by Census Bureau field representatives across the country through both personal
    and telephone interviews. CPS collects labor force data, such as annual work
    activity and income, veteran status, school enrollment, contingent employment,
    worker displacement, job tenure, and more.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The survey date year (format YYYY) to include in the dataset. Will
        return an empty DataFrame if there are no data with this year. If None is
        provided, data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :param engine:
        Engine to use for loading data. Determines the return type.
    :return: A DataFrame of simulated CPS data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.extend(
            [
                (DATASETS.cps.date_column_name, ">=", pd.Timestamp(f"{year}-01-01")),
                (DATASETS.cps.date_column_name, "<=", pd.Timestamp(f"{year}-12-31")),
            ]
        )
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.cps.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.cps, source, seed, config, user_filters, verbose, engine=engine
    )


def generate_taxes_w2_and_1099(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "modin"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople W2 and 1099 tax dataset which represents simulated
    tax form data.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The tax year (format YYYY) to include in the dataset. Will return
        an empty DataFrame if there are no data with this year. If None is provided,
        data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :param engine:
        Engine to use for loading data. Determines the return type.
    :return: A DataFrame of simulated W2 and 1099 tax data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append((DATASETS.tax_w2_1099.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.tax_w2_1099.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.tax_w2_1099, source, seed, config, user_filters, verbose, engine=engine
    )


def generate_women_infants_and_children(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "modin"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople WIC dataset which represents a simulated version of
    the administrative data that would be recorded by WIC. This is a yearly file
    of information about all simulants enrolled in the program as of the end of that year.

    The Special Supplemental Nutrition Program for Women, Infants, and Children (WIC)
    is a government benefits program designed to support mothers and young children.
    The main qualifications are income and the presence of young children in the home.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The year (format YYYY) to include in the dataset. Will return an
        empty DataFrame if there are no data with this year. If None is provided,
        data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :param engine:
        Engine to use for loading data. Determines the return type.
    :return: A DataFrame of simulated WIC data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append((DATASETS.wic.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.wic.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.wic, source, seed, config, user_filters, verbose, engine=engine
    )


def generate_social_security(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    verbose: bool = False,
    engine: Literal["pandas", "modin"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople SSA dataset which represents simulated Social Security
    Administration (SSA) data.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The latest year (format YYYY) to include in the dataset; will also
        include all previous years. Will return an empty DataFrame if there are no
        data on or before this year. If None is provided, data from all years are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :param engine:
        Engine to use for loading data. Determines the return type.
    :return: A DataFrame of simulated SSA data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append(
            (DATASETS.ssa.date_column_name, "<=", pd.Timestamp(f"{year}-12-31"))
        )
        seed = seed * 10_000 + year
    return _generate_dataset(
        DATASETS.ssa, source, seed, config, user_filters, verbose, engine=engine
    )


def generate_taxes_1040(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "modin"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople 1040 tax dataset which represents simulated
    tax form data.

    :param source: The root directory containing pseudopeople input data. Defaults
        to the pseudopeople sample datasets directory.
    :param seed: An integer seed for randomness. Defaults to 0.
    :param config: An optional override to the default configuration. Can be a path
        to a configuration YAML file or a dictionary.
    :param year: The tax year (format YYYY) to include in the dataset. Will return
        an empty DataFrame if there are no data with this year. If None is provided,
        data from all years are included in the dataset.
    :param state: The state string to include in the dataset. Either full name or
        abbreviation (e.g., "Ohio" or "OH"). Will return an empty DataFrame if there are no
        data pertaining to this state. If None is provided, data from all locations are
        included in the dataset.
    :param verbose: Log with verbosity if True.
    :param engine:
        Engine to use for loading data. Determines the return type.
    :return: A DataFrame of simulated 1040 tax data.
    :raises ConfigurationError: An incorrect config is provided.
    :raises DataSourceError: An incorrect pseudopeople input data source is provided.
    """
    user_filters = []
    if year:
        user_filters.append((DATASETS.tax_1040.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state:
        user_filters.append(
            (DATASETS.tax_1040.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.tax_1040, source, seed, config, user_filters, verbose, engine=engine
    )


def fetch_filepaths(dataset: Dataset, source: Path) -> Union[List, List[dict]]:
    # returns a list of filepaths for all Datasets
    data_paths = get_dataset_filepaths(source, dataset.name)

    return data_paths


def validate_data_path_suffix(data_paths) -> None:
    suffix = set(x.suffix for x in data_paths)
    if len(suffix) > 1:
        raise DataSourceError(
            f"Only one type of file extension expected but more than one found: {suffix}. "
            "Please provide the path to the unmodified root data directory."
        )

    return None


def get_dataset_filepaths(source: Path, dataset_name: str) -> List[Path]:
    directory = source / dataset_name
    dataset_paths = [x for x in directory.glob(f"{dataset_name}*")]
    sorted_dataset_paths = sorted(dataset_paths)
    return sorted_dataset_paths
