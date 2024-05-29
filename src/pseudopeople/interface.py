from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from layered_config_tree import LayeredConfigTree
from loguru import logger
from packaging.version import parse
from tqdm import tqdm

from pseudopeople import __version__ as psp_version
from pseudopeople.configuration import get_configuration
from pseudopeople.constants import paths
from pseudopeople.constants.metadata import DATEFORMATS
from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS
from pseudopeople.dtypes import DtypeNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.loader import load_standard_dataset
from pseudopeople.noise import noise_dataset
from pseudopeople.schema_entities import COLUMNS, DATASETS, Dataset
from pseudopeople.utilities import (
    PANDAS_ENGINE,
    DataFrame,
    configure_logging_to_terminal,
    get_engine_from_string,
    get_state_abbreviation,
    to_string,
)


def _generate_dataset(
    dataset: Dataset,
    source: Union[Path, str],
    seed: int,
    config: Union[Path, str, Dict],
    user_filters: List[tuple],
    verbose: bool = False,
    engine_name: Literal["pandas", "dask"] = "pandas",
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
    :param engine_name:
        String indicating engine to use for loading data. Determines the return type.
    :return:
        Noised dataset data in a dataframe
    """
    configure_logging_to_terminal(verbose)
    configuration_tree = get_configuration(config, dataset, user_filters)

    if source is None:
        source = paths.SAMPLE_DATA_ROOT
    else:
        source = Path(source)
        validate_source_compatibility(source, dataset)

    engine = get_engine_from_string(engine_name)

    if engine == PANDAS_ENGINE:
        # We process shards serially
        data_file_paths = fetch_filepaths(dataset, source)
        if not data_file_paths:
            raise DataSourceError(
                f"No datasets found at directory {str(source)}. "
                "Please provide the path to the unmodified root data directory."
            )

        validate_data_path_suffix(data_file_paths)

        # Iterate sequentially
        noised_dataset = []
        iterator = (
            tqdm(data_file_paths, desc="Noising data", leave=False)
            if len(data_file_paths) > 1
            else data_file_paths
        )

        for data_file_index, data_file_path in enumerate(iterator):
            logger.debug(f"Loading data from {data_file_path}.")
            data = load_standard_dataset(
                data_file_path, user_filters, engine=engine, is_file=True
            )
            if len(data.index) == 0:
                continue
            # Use a different seed for each data file/shard, otherwise the randomness will duplicate
            # and the Nth row in each shard will get the same noise
            data_path_seed = f"{seed}_{data_file_index}"
            noised_data = _prep_and_noise_dataset(
                data, dataset, configuration_tree, data_path_seed
            )
            noised_dataset.append(noised_data)

        # Check if all shards for the dataset are empty
        if len(noised_dataset) == 0:
            raise ValueError(
                "Invalid value provided for 'state' or 'year'. No data found with "
                f"the user provided 'state' or 'year' filters at {source / dataset.name}."
            )
        noised_dataset = pd.concat(noised_dataset, ignore_index=True)

        noised_dataset = _coerce_dtypes(
            noised_dataset,
            dataset,
        )
    else:
        try:
            from distributed.client import default_client

            default_client().run(lambda: configure_logging_to_terminal(verbose))
        except (ImportError, ValueError):
            # Not using a distributed cluster, so the configure_logging_to_terminal call above already did everything
            pass

        # Let dask deal with how to partition the shards -- we pass it the
        # entire directory containing the parquet files
        data_directory_path = source / dataset.name
        import dask

        # Our work depends on the particulars of how dtypes work, and is only
        # built to work with NumPy dtypes, so we turn off the Dask default behavior
        # of using PyArrow dtypes.
        with dask.config.set({"dataframe.convert-string": False}):
            data = load_standard_dataset(
                data_directory_path, user_filters, engine=engine, is_file=False
            )
            # We are about to check the length, which requires computation anyway, so we cache
            # that computation
            data = data.persist()

            # Check if all shards for the dataset are empty
            if len(data) == 0:
                raise ValueError(
                    "Invalid value provided for 'state' or 'year'. No data found with "
                    f"the user provided 'state' or 'year' filters at {data_directory_path}."
                )

            noised_dataset = data.map_partitions(
                lambda df, partition_info=None: _coerce_dtypes(
                    _prep_and_noise_dataset(
                        df,
                        dataset,
                        configuration_tree,
                        seed=f"{seed}_{partition_info['number'] if partition_info is not None else 1}",
                        progress_bar=False,
                    ),
                    dataset,
                ),
                meta=[(c.name, c.dtype_name) for c in dataset.columns],
            )

    logger.debug("*** Finished ***")

    return noised_dataset


def _prep_and_noise_dataset(
    data: pd.DataFrame,
    dataset: Dataset,
    configuration_tree: LayeredConfigTree,
    seed: Any,
    progress_bar: bool = True,
) -> pd.DataFrame:
    data = _reformat_dates_for_noising(data, dataset)
    data = _clean_input_data(data, dataset)
    noised_data = noise_dataset(
        dataset, data, configuration_tree, seed, progress_bar=progress_bar
    )
    noised_data = _extract_columns(dataset.columns, noised_data)
    return noised_data


def validate_source_compatibility(source: Path, dataset: Dataset):
    # TODO [MIC-4546]: Clean this up w/ metadata and update test_interface.py tests to be generic
    directories = [x.name for x in source.iterdir() if x.is_dir()]
    if dataset.name not in directories:
        raise FileNotFoundError(
            f"Could not find '{dataset.name}' in '{source}'. Please check that the provided source "
            "directory is correct. If using the sample data, no source is required. If providing a source, "
            f"a directory should provided that has a subdirectory for '{dataset.name}'. "
        )
    changelog = source / "CHANGELOG.rst"
    if changelog.exists():
        version = _get_data_changelog_version(changelog)
        if version > parse("1.4.2"):
            raise DataSourceError(
                f"The provided simulated population data is incompatible with this version of pseudopeople ({psp_version}).\n"
                "A newer version of simulated population data has been provided.\n"
                "Please upgrade the pseudopeople package."
            )
        if version < parse("1.4.2"):
            raise DataSourceError(
                f"The provided simulated population data is incompatible with this version of pseudopeople ({psp_version}).\n"
                "The simulated population data has been corrupted.\n"
                "Please re-download the simulated population data."
            )
    else:
        raise DataSourceError(
            f"The provided simulated population data is incompatible with this version of pseudopeople ({psp_version}).\n"
            "An older version of simulated population data has been provided.\n"
            "Please either request updated simulated population data or downgrade the pseudopeople package."
        )


def _get_data_changelog_version(changelog):
    with open(changelog, "r") as file:
        first_line = file.readline()
    version = parse(first_line.split("**")[1].split("-")[0].strip())
    return version


def _clean_input_data(
    data: pd.DataFrame,
    dataset: Dataset,
) -> pd.DataFrame:
    for col in dataset.columns:
        # Coerce empty strings to nans
        data[col.name] = data[col.name].replace("", np.nan)

        if data[col.name].dtype.name == "category" and col.dtype_name == DtypeNames.OBJECT:
            # We made some columns in the pseudopeople input categorical
            # purely as a kind of DIY compression.
            # TODO: Determine whether this is benefitting us after
            # the switch to Parquet.
            data[col.name] = to_string(data[col.name])

    return data


def _coerce_dtypes(
    data: pd.DataFrame,
    dataset: Dataset,
) -> pd.DataFrame:
    for col in dataset.columns:
        if col.dtype_name != data[col.name].dtype.name:
            if col.dtype_name == DtypeNames.OBJECT:
                data[col.name] = to_string(data[col.name])
            else:
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
                # Avoid running strftime on large data, since that will
                # re-parse the format string for each row
                # https://github.com/pandas-dev/pandas/issues/44764
                # Year is already guaranteed to be 4-digit: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-timestamp-limits
                is_na = data[column].isna()
                data_column = data.loc[~is_na, column]
                year_string = data_column.dt.year.astype(str)
                month_string = _zfill_fast(data_column.dt.month.astype(str), 2)
                day_string = _zfill_fast(data_column.dt.day.astype(str), 2)
                if dataset.date_format == DATEFORMATS.YYYYMMDD:
                    result = year_string + month_string + day_string
                elif dataset.date_format == DATEFORMATS.MM_DD_YYYY:
                    result = month_string + "/" + day_string + "/" + year_string
                elif dataset.date_format == DATEFORMATS.MMDDYYYY:
                    result = month_string + day_string + year_string
                else:
                    raise ValueError(f"Invalid date format in {dataset.name}.")

                data[column] = pd.Series(np.nan, dtype=str)
                data.loc[~is_na, column] = result

    return data


def _zfill_fast(col: pd.Series, desired_length: int) -> pd.Series:
    """Performs the same operation as col.str.zfill(desired_length), but vectorized."""
    # The most zeroes that could ever be needed would be desired_length
    maximum_padding = ("0" * desired_length) + col
    # Now trim to only the zeroes needed
    return maximum_padding.str[-desired_length:]


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
    engine: Literal["pandas", "dask"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople decennial census dataset which represents
    simulated responses to the US Census Bureau's Census of Population
    and Housing.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The year for which to generate a simulated decennial census of
        the simulated population (format YYYY, e.g., 2030). Must be a
        decennial year (e.g., 2020, 2030, 2040). Default is 2020. If
        `None` is passed instead, data for all available years are
        included in the returned dataset.

    :param state:

        The US state for which to generate a simulated census of the
        simulated population, or `None` (default) to generate data for
        all available US states. The returned dataset will contain data
        for simulants living in the specified state on Census Day (April
        1) of the specified year. Can be a full state name or a state
        abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :param engine:

        Engine to use for loading data. Determines the return type.
        Default is "pandas" which returns a pandas DataFrame.
        "dask" returns a Dask DataFrame and requires Dask to be
        installed (e.g. `pip install pseudopeople[dask]`).
        It runs the dataset generation on a Dask cluster, which can
        parallelize and run out-of-core.

    :return:

        A DataFrame of simulated decennial census data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        user_filters.append((DATASETS.census.date_column_name, "==", year))
    if state is not None:
        user_filters.append(
            (DATASETS.census.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.census, source, seed, config, user_filters, verbose, engine_name=engine
    )


def generate_american_community_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "dask"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople ACS dataset which represents simulated
    responses to the ACS survey.

    The American Community Survey (ACS) is an ongoing household survey
    conducted by the US Census Bureau that gathers information on a
    rolling basis about American community populations. Information
    collected includes ancestry, citizenship, education, income,
    language proficiency, migration, employment, disability, and housing
    characteristics.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The year for which to generate simulated American Community
        Surveys of the simulated population (format YYYY, e.g., 2036);
        the simulated dataset will contain records for surveys conducted
        on any date in the specified year. Default is 2020. If `None` is
        passed instead, data for all available years are included in the
        returned dataset.

    :param state:

        The US state for which to generate simulated American Community
        Surveys of the simulated population, or `None` (default) to
        generate data for all available US states. The returned dataset
        will contain survey data for simulants living in the specified
        state during the specified year. Can be a full state name or a
        state abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :param engine:

        Engine to use for loading data. Determines the return type.
        Default is "pandas" which returns a pandas DataFrame.
        "dask" returns a Dask DataFrame and requires Dask to be
        installed (e.g. `pip install pseudopeople[dask]`).
        It runs the dataset generation on a Dask cluster, which can
        parallelize and run out-of-core.

    :return:

        A DataFrame of simulated ACS data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        try:
            user_filters.extend(
                [
                    (
                        DATASETS.acs.date_column_name,
                        ">=",
                        pd.Timestamp(year=year, month=1, day=1),
                    ),
                    (
                        DATASETS.acs.date_column_name,
                        "<=",
                        pd.Timestamp(year=year, month=12, day=31),
                    ),
                ]
            )
        except (pd.errors.OutOfBoundsDatetime, ValueError):
            raise ValueError(f"Invalid year provided: '{year}'")
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.extend(
            [(DATASETS.acs.state_column_name, "==", get_state_abbreviation(state))]
        )
    return _generate_dataset(
        DATASETS.acs, source, seed, config, user_filters, verbose, engine_name=engine
    )


def generate_current_population_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "dask"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople CPS dataset which represents simulated
    responses to the CPS survey.

    The Current Population Survey (CPS) is a household survey conducted
    by the US Census Bureau and the US Bureau of Labor Statistics. This
    survey is administered by Census Bureau field representatives across
    the country through both personal and telephone interviews. CPS
    collects labor force data, such as annual work activity and income,
    veteran status, school enrollment, contingent employment, worker
    displacement, job tenure, and more.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The year for which to generate simulated Current Population
        Surveys of the simulated population (format YYYY, e.g., 2036);
        the simulated dataset will contain records for surveys conducted
        on any date in the specified year. Default is 2020. If `None` is
        passed instead, data for all available years are included in the
        returned dataset.

    :param state:

        The US state for which to generate simulated Current Population
        Surveys of the simulated population, or `None` (default) to
        generate data for all available US states. The returned dataset
        will contain survey data for simulants living in the specified
        state during the specified year. Can be a full state name or a
        state abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :param engine:

        Engine to use for loading data. Determines the return type.
        Default is "pandas" which returns a pandas DataFrame.
        "dask" returns a Dask DataFrame and requires Dask to be
        installed (e.g. `pip install pseudopeople[dask]`).
        It runs the dataset generation on a Dask cluster, which can
        parallelize and run out-of-core.

    :return:

        A DataFrame of simulated CPS data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        try:
            user_filters.extend(
                [
                    (
                        DATASETS.cps.date_column_name,
                        ">=",
                        pd.Timestamp(year=year, month=1, day=1),
                    ),
                    (
                        DATASETS.cps.date_column_name,
                        "<=",
                        pd.Timestamp(year=year, month=12, day=31),
                    ),
                ]
            )
        except (pd.errors.OutOfBoundsDatetime, ValueError):
            raise ValueError(f"Invalid year provided: '{year}'")
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.extend(
            [(DATASETS.cps.state_column_name, "==", get_state_abbreviation(state))]
        )
    return _generate_dataset(
        DATASETS.cps, source, seed, config, user_filters, verbose, engine_name=engine
    )


def generate_taxes_w2_and_1099(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "dask"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople W2 and 1099 tax dataset which represents
    simulated tax form data.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The tax year for which to generate records (format YYYY, e.g.,
        2036); the simulated dataset will contain the W2 & 1099 tax
        forms filed by simulated employers for the specified year.
        Default is 2020. If `None` is passed instead, data for all
        available years are included in the returned dataset.

    :param state:

        The US state for which to generate tax records from the
        simulated population, or `None` (default) to generate data for
        all available US states. The returned dataset will contain W2 &
        1099 tax forms filed for simulants living in the specified state
        during the specified tax year. Can be a full state name or a
        state abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :param engine:

        Engine to use for loading data. Determines the return type.
        Default is "pandas" which returns a pandas DataFrame.
        "dask" returns a Dask DataFrame and requires Dask to be
        installed (e.g. `pip install pseudopeople[dask]`).
        It runs the dataset generation on a Dask cluster, which can
        parallelize and run out-of-core.

    :return:

        A DataFrame of simulated W2 and 1099 tax data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        user_filters.append((DATASETS.tax_w2_1099.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.append(
            (DATASETS.tax_w2_1099.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.tax_w2_1099, source, seed, config, user_filters, verbose, engine_name=engine
    )


def generate_women_infants_and_children(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "dask"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople WIC dataset which represents a simulated
    version of the administrative data that would be recorded by WIC.
    This is a yearly file of information about all simulants enrolled in
    the program as of the end of that year.

    The Special Supplemental Nutrition Program for Women, Infants, and
    Children (WIC) is a government benefits program designed to support
    mothers and young children. The main qualifications are income and
    the presence of young children in the home.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The year for which to generate WIC administrative records
        (format YYYY, e.g., 2036); the simulated dataset will contain
        records for simulants enrolled in WIC at the end of the
        specified year (or on May 1, 2041 if `year=2041` since that is
        the end date of the simulation). Default is 2020. If `None` is
        passed instead, data for all available years are included in the
        returned dataset.

    :param state:

        The US state for which to generate WIC administrative records
        from the simulated population, or `None` (default) to generate
        data for all available US states. The returned dataset will
        contain records for enrolled simulants living in the specified
        state at the end of the specified year (or on May 1, 2041 if
        `year=2041` since that is the end date of the simulation). Can
        be a full state name or a state abbreviation (e.g., "Ohio" or
        "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :param engine:

        Engine to use for loading data. Determines the return type.
        Default is "pandas" which returns a pandas DataFrame.
        "dask" returns a Dask DataFrame and requires Dask to be
        installed (e.g. `pip install pseudopeople[dask]`).
        It runs the dataset generation on a Dask cluster, which can
        parallelize and run out-of-core.

    :return:

        A DataFrame of simulated WIC data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        user_filters.append((DATASETS.wic.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.append(
            (DATASETS.wic.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.wic, source, seed, config, user_filters, verbose, engine_name=engine
    )


def generate_social_security(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    verbose: bool = False,
    engine: Literal["pandas", "dask"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople SSA dataset which represents simulated
    Social Security Administration (SSA) data.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The final year of simulated social security records to include
        in the dataset (format YYYY, e.g., 2036); will also include
        records from all previous years. Default is 2020. If `None` is
        passed instead, data for all available years are included in the
        returned dataset.

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :param engine:

        Engine to use for loading data. Determines the return type.
        Default is "pandas" which returns a pandas DataFrame.
        "dask" returns a Dask DataFrame and requires Dask to be
        installed (e.g. `pip install pseudopeople[dask]`).
        It runs the dataset generation on a Dask cluster, which can
        parallelize and run out-of-core.

    :return:

        A DataFrame of simulated SSA data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or any prior years.
    """
    user_filters = []
    if year is not None:
        try:
            user_filters.append(
                (
                    DATASETS.ssa.date_column_name,
                    "<=",
                    pd.Timestamp(year=year, month=12, day=31),
                )
            )
        except (pd.errors.OutOfBoundsDatetime, ValueError):
            raise ValueError(f"Invalid year provided: '{year}'")
        seed = seed * 10_000 + year
    return _generate_dataset(
        DATASETS.ssa, source, seed, config, user_filters, verbose, engine_name=engine
    )


def generate_taxes_1040(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, Dict[str, Dict]] = None,
    year: Optional[int] = 2020,
    state: Optional[str] = None,
    verbose: bool = False,
    engine: Literal["pandas", "dask"] = "pandas",
) -> DataFrame:
    """
    Generates a pseudopeople 1040 tax dataset which represents simulated
    tax form data.

    :param source:

        The root directory containing pseudopeople simulated population
        data. Defaults to using the included sample population when
        source is `None`.

    :param seed:

        An integer seed for randomness. Defaults to 0.

    :param config:

        An optional override to the default configuration. Can be a path
        to a configuration YAML file, a configuration dictionary, or the
        sentinel value `pseudopeople.NO_NOISE`, which will generate a
        dataset without any configurable noise.

    :param year:

        The tax year for which to generate records (format YYYY, e.g.,
        2036); the simulated dataset will contain the 1040 tax forms
        filed by simulants for the specified year. Default is 2020. If
        `None` is passed instead, data for all available years are
        included in the returned dataset.

    :param state:

        The US state for which to generate tax records from the
        simulated population, or `None` (default) to generate data for
        all available US states. The returned dataset will contain 1040
        tax forms filed by simulants living in the specified state
        during the specified tax year. Can be a full state name or a
        state abbreviation (e.g., "Ohio" or "OH").

    :param verbose:

        Log with verbosity if `True`. Default is `False`.

    :param engine:

        Engine to use for loading data. Determines the return type.
        Default is "pandas" which returns a pandas DataFrame.
        "dask" returns a Dask DataFrame and requires Dask to be
        installed (e.g. `pip install pseudopeople[dask]`).
        It runs the dataset generation on a Dask cluster, which can
        parallelize and run out-of-core.

    :return:

        A DataFrame of simulated 1040 tax data.

    :raises ConfigurationError:

        An invalid `config` is provided.

    :raises DataSourceError:

        An invalid pseudopeople simulated population data source is
        provided.

    :raises ValueError:

        The simulated population has no data for this dataset in the
        specified year or state.
    """
    user_filters = []
    if year is not None:
        user_filters.append((DATASETS.tax_1040.date_column_name, "==", year))
        seed = seed * 10_000 + year
    if state is not None:
        user_filters.append(
            (DATASETS.tax_1040.state_column_name, "==", get_state_abbreviation(state))
        )
    return _generate_dataset(
        DATASETS.tax_1040, source, seed, config, user_filters, verbose, engine_name=engine
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
