from pathlib import Path
from typing import Dict, Union

import pandas as pd
import pyarrow.parquet as pq
from loguru import logger

from pseudopeople.configuration import get_configuration
from pseudopeople.constants import paths
from pseudopeople.noise import noise_dataset
from pseudopeople.schema_entities import COLUMNS, DATASETS, Dataset
from pseudopeople.utilities import configure_logging_to_terminal


def _generate_dataset(
    dataset: Dataset,
    source: Union[Path, str],
    seed: int,
    config: Union[Path, str, dict],
    year_filter: Dict,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Helper for generating noised datasets from clean data.

    :param dataset:
        Dataset needing to be noised
    :param source:
        Root directory of clean data input which needs to be noised
    :param seed:
        Seed for controlling randomness
    :param config:
        Object to configure noise levels
    :param year_filter:
        Dictionary with keys 'hdf' and 'parquet' and values filter lists
    :param verbose:
        Log with verbosity if True. Default is False.
    :return:
        Noised dataset data in a pd.DataFrame
    """
    configure_logging_to_terminal(verbose)
    configuration_tree = get_configuration(config)
    # TODO: we should save outputs of the simulation with filenames that are
    #  consistent with the names of the datasets if possible.
    dataset_file_name = {
        DATASETS.acs.name: "household_survey_observer_acs",
        DATASETS.cps.name: "household_survey_observer_cps",
        DATASETS.tax_w2_1099.name: "tax_w2_observer",
        DATASETS.wic.name: "wic_observer",
    }.get(dataset.name, f"{dataset.name}_observer")
    if source is None:
        source = paths.SAMPLE_DATA_ROOT
    source = Path(source) / dataset_file_name
    data_paths = [x for x in source.glob(f"{dataset_file_name}*")]
    if not data_paths:
        raise ValueError(
            f"No datasets found at directory {str(source)}. "
            "Please provide the path to the unmodified root data directory."
        )
    suffix = set(x.suffix for x in data_paths)
    if len(suffix) > 1:
        raise TypeError(
            f"Only one type of file extension expected but more than one found: {suffix}. "
            "Please provide the path to the unmodified root data directory."
        )
    noised_dataset = []
    for data_path in data_paths:
        logger.info(f"Loading data from {data_path}.")
        data = _load_data_from_path(data_path, year_filter)

        data = _reformat_dates_for_noising(data, dataset)
        data = _coerce_dtypes(data, dataset)
        noised_data = noise_dataset(dataset, data, configuration_tree, seed)
        noised_data = _extract_columns(dataset.columns, noised_data)
        noised_dataset.append(noised_data)

    noised_dataset = pd.concat(noised_dataset, ignore_index=True)

    # Known pandas bug: pd.concat does not preserve category dtypes so we coerce
    # again after concat (https://github.com/pandas-dev/pandas/issues/51362)
    noised_dataset = _coerce_dtypes(noised_dataset, dataset)

    return noised_dataset


def _coerce_dtypes(data: pd.DataFrame, dataset: Dataset):
    # Coerce dtypes prior to noising to catch issues early as well as
    # get most columns away from dtype 'category' and into 'object' (strings)
    for col in dataset.columns:
        if col.dtype_name != data[col.name].dtype.name:
            data[col.name] = data[col.name].astype(col.dtype_name)
    return data


def _load_data_from_path(data_path: Path, year_filter: Dict):
    """Load data from a data file given a data_path and a year_filter."""
    if data_path.suffix == ".hdf":
        with pd.HDFStore(str(data_path), mode="r") as hdf_store:
            data = hdf_store.select("data", where=year_filter["hdf"])
    elif data_path.suffix == ".parquet":
        data = pq.read_table(data_path, filters=year_filter["parquet"]).to_pandas()
    else:
        raise ValueError(
            "Source path must either be a .hdf or a .parquet file. Provided "
            f"{data_path.suffix}"
        )
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"File located at {data_path} must contain a pandas DataFrame. "
            "Please provide the path to the unmodified root data directory."
        )
    return data


def _reformat_dates_for_noising(data: pd.DataFrame, dataset: Dataset):
    """Formats SSA event_date and dates of birth, so they can be noised."""
    if COLUMNS.ssa_event_date.name in data.columns and dataset == DATASETS.ssa:
        # event_date -> YYYYMMDD
        data[COLUMNS.ssa_event_date.name] = data[COLUMNS.ssa_event_date.name].dt.strftime(
            "%Y%m%d"
        )
    if COLUMNS.dob.name in data.columns:
        # date_of_birth -> MM/DD/YYYY
        data[COLUMNS.dob.name] = data[COLUMNS.dob.name].dt.strftime("%m/%d/%Y")
    return data


def _extract_columns(columns_to_keep, noised_dataset):
    """Helper function for test mocking purposes"""
    if columns_to_keep:
        noised_dataset = noised_dataset[[c.name for c in columns_to_keep]]
    return noised_dataset


def generate_decennial_census(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, dict] = None,
    year: int = 2020,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates noised decennial census data from un-noised data.

    :param source: A path to un-noised source census data
    :param seed: An integer seed for randomness
    :param config: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :param year: The year from the data to noise
    :param verbose: Log with verbosity if True. Default is False.
    :return: A pd.DataFrame of noised census data
    """
    year_filter = {"hdf": None, "parquet": None}
    if year:
        year_filter["hdf"] = [f"{DATASETS.census.date_column} == {year}."]
        year_filter["parquet"] = [(DATASETS.census.date_column, "==", year)]
    return _generate_dataset(DATASETS.census, source, seed, config, year_filter, verbose)


def generate_american_community_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, dict] = None,
    year: int = 2020,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates noised American Community Survey (ACS) data from un-noised data.

    :param source: A path to un-noised source ACS data
    :param seed: An integer seed for randomness
    :param config: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :param year: The year from the data to noise
    :param verbose: Log with verbosity if True. Default is False.
    :return: A pd.DataFrame of noised ACS data
    """
    year_filter = {"hdf": None, "parquet": None}
    if year:
        year_filter["hdf"] = [
            f"{DATASETS.acs.date_column} >= '{year}-01-01' and {DATASETS.acs.date_column} <= '{year}-12-31'"
        ]
        year_filter["parquet"] = [
            (DATASETS.acs.date_column, ">=", pd.Timestamp(f"{year}-01-01")),
            (DATASETS.acs.date_column, "<=", pd.Timestamp(f"{year}-12-31")),
        ]
        seed = seed * 10_000 + year
    return _generate_dataset(DATASETS.acs, source, seed, config, year_filter, verbose)


def generate_current_population_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, dict] = None,
    year: int = 2020,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates noised Current Population Survey (CPS) data from un-noised data.

    :param source: A path to un-noised source CPS data
    :param seed: An integer seed for randomness
    :param config: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :param year: The year from the data to noise
    :param verbose: Log with verbosity if True. Default is False.
    :return: A pd.DataFrame of noised CPS data
    """
    year_filter = {"hdf": None, "parquet": None}
    if year:
        year_filter["hdf"] = [
            f"{DATASETS.cps.date_column} >= '{year}-01-01' and {DATASETS.cps.date_column} <= '{year}-12-31'"
        ]
        year_filter["parquet"] = [
            (DATASETS.cps.date_column, ">=", pd.Timestamp(f"{year}-01-01")),
            (DATASETS.cps.date_column, "<=", pd.Timestamp(f"{year}-12-31")),
        ]
        seed = seed * 10_000 + year
    return _generate_dataset(DATASETS.cps, source, seed, config, year_filter, verbose)


def generate_taxes_w2_and_1099(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, dict] = None,
    year: int = 2020,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates noised W2 and 1099 data from un-noised data.

    :param source: A path to un-noised source W2 and 1099 data
    :param seed: An integer seed for randomness
    :param config: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :param year: The year from the data to noise
    :param verbose: Log with verbosity if True. Default is False.
    :return: A pd.DataFrame of noised W2 and 1099 data
    """
    year_filter = {"hdf": None, "parquet": None}
    if year:
        year_filter["hdf"] = [f"{DATASETS.tax_w2_1099.date_column} == {year}."]
        year_filter["parquet"] = [(DATASETS.tax_w2_1099.date_column, "==", year)]
        seed = seed * 10_000 + year
    return _generate_dataset(DATASETS.tax_w2_1099, source, seed, config, year_filter, verbose)


def generate_women_infants_and_children(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, dict] = None,
    year: int = 2020,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates noised Women Infants and Children (WIC) data from un-noised data.

    :param source: A path to un-noised source WIC data
    :param seed: An integer seed for randomness
    :param config: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :param year: The year from the data to noise
    :param verbose: Log with verbosity if True. Default is False.
    :return: A pd.DataFrame of noised WIC data
    """
    year_filter = {"hdf": None, "parquet": None}
    if year:
        year_filter["hdf"] = [f"{DATASETS.wic.date_column} == {year}."]
        year_filter["parquet"] = [(DATASETS.wic.date_column, "==", year)]
        seed = seed * 10_000 + year
    return _generate_dataset(DATASETS.wic, source, seed, config, year_filter, verbose)


def generate_social_security(
    source: Union[Path, str] = None,
    seed: int = 0,
    config: Union[Path, str, dict] = None,
    year: int = 2020,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates noised Social Security (SSA) data from un-noised data.

    :param source: A path to un-noised source SSA data
    :param seed: An integer seed for randomness
    :param config: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :param year: The year up to which to noise from the data
    :param verbose: Log with verbosity if True. Default is False.
    :return: A pd.DataFrame of noised SSA data
    """
    year_filter = {"hdf": None, "parquet": None}
    if year:
        year_filter["hdf"] = [f"{DATASETS.ssa.date_column} <= {year}."]
        year_filter["parquet"] = [
            (DATASETS.ssa.date_column, "<=", pd.Timestamp(f"{year}-12-31"))
        ]
        seed = seed * 10_000 + year
    return _generate_dataset(DATASETS.ssa, source, seed, config, year_filter, verbose)
