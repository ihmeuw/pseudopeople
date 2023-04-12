from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger

from pseudopeople.configuration import get_configuration
from pseudopeople.constants import paths
from pseudopeople.noise import noise_form
from pseudopeople.schema_entities import FORMS, Form


def _generate_form(
    form: Form,
    source: Union[Path, str],
    seed: int,
    configuration: Union[Path, str, dict],
) -> pd.DataFrame:
    """
    Helper for generating noised forms from clean data.

    :param form:
        Form needing to be noised
    :param source:
        Root directory of clean data input which needs to be noised
    :param seed:
        Seed for controlling randomness
    :param configuration:
        Object to configure noise levels
    :return:
        Noised form data in a pd.DataFrame
    """
    configuration_tree = get_configuration(configuration)
    # TODO: we should save outputs of the simulation with filenames that are
    #  consistent with the names of the forms if possible.
    form_file_name = {
        FORMS.acs.name: "household_survey_observer_acs",
        FORMS.cps.name: "household_survey_observer_cps",
        FORMS.tax_w2_1099.name: "tax_w2_observer",
        FORMS.wic.name: "wic_observer",
    }.get(form.name, f"{form.name}_observer")
    if source is None:
        source = paths.SAMPLE_DATA_ROOT
    source = Path(source) / form_file_name
    data_paths = [x for x in source.glob(f"{form_file_name}*")]
    if not data_paths:
        logger.warning(
            f"No datasets found at directory {str(source)}. "
            "Please provide the path to the unmodified root data directory."
        )
        return None
    suffix = set(x.suffix for x in data_paths)
    if len(suffix) > 1:
        raise TypeError(
            f"Only one type of file extension expected but more than one found: {suffix}. "
            "Please provide the path to the unmodified root data directory."
        )
    noised_form = []
    for data_path in data_paths:
        if data_path.suffix == ".hdf":
            data = pd.read_hdf(data_path)
        elif data_path.suffix == ".parquet":
            data = pd.read_parquet(data_path)
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

        columns_to_keep = [c for c in form.columns]

        # Coerce dtypes
        for col in columns_to_keep:
            if col.dtype_name != data[col.name].dtype.name:
                data[col.name] = data[col.name].astype(col.dtype_name)

        noised_data = noise_form(form, data, configuration_tree, seed)
        noised_form.append(noised_data[[c.name for c in columns_to_keep]])

    return pd.concat(noised_form, ignore_index=True)


# TODO: add year as parameter to select the year of the decennial census to generate (MIC-3909)
def generate_decennial_census(
    source: Union[Path, str] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised decennial census data from un-noised data.

    :param source: A path to un-noised source census data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised census data
    """
    return _generate_form(FORMS.census, source, seed, configuration)


def generate_american_communities_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised American Communities Survey (ACS) data from un-noised data.

    :param source: A path to un-noised source ACS data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised ACS data
    """
    return _generate_form(FORMS.acs, source, seed, configuration)


def generate_current_population_survey(
    source: Union[Path, str] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised Current Population Survey (CPS) data from un-noised data.

    :param source: A path to un-noised source CPS data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised CPS data
    """
    return _generate_form(FORMS.cps, source, seed, configuration)


def generate_taxes_w2_and_1099(
    source: Union[Path, str] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised W2 and 1099 data from un-noised data.

    :param source: A path to un-noised source W2 and 1099 data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised W2 and 1099 data
    """
    return _generate_form(FORMS.tax_w2_1099, source, seed, configuration)


def generate_women_infants_and_children(
    source: Union[Path, str] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised Women Infants and Children (WIC) data from un-noised data.

    :param source: A path to un-noised source WIC data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised WIC data
    """
    return _generate_form(FORMS.wic, source, seed, configuration)


def generate_social_security(
    source: Union[Path, str] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised Social Security (SSA) data from un-noised data.

    :param source: A path to un-noised source SSA data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised SSA data
    """
    return _generate_form(FORMS.ssa, source, seed, configuration)
