from pathlib import Path
from typing import Union

import pandas as pd

from pseudopeople.configuration import get_configuration
from pseudopeople.constants import paths
from pseudopeople.noise import noise_form
from pseudopeople.schema_entities import FORMS, Form


def _generate_form(
    form: Form,
    source: Union[Path, str, pd.DataFrame],
    seed: int,
    configuration: Union[Path, str, dict],
) -> pd.DataFrame:
    """
    Helper for generating noised forms from clean data.

    :param form:
        Form needing to be noised
    :param source:
        Clean data input which needs to be noised
    :param seed:
        Seed for controlling randomness
    :param configuration:
        Object to configure noise levels
    :return:
        Noised form data in a pd.DataFrame
    """
    configuration_tree = get_configuration(configuration)
    if source is None:
        # TODO: hard-coding the .parquet extension for now. This will go away
        #  once we only support passing the root directory of the data.
        # TODO: we should save outputs of the simulation with filenames that are
        #  consistent with the names of the forms if possible.
        form_file_name = {
            FORMS.acs.name: "household_survey_observer_acs",
            FORMS.cps.name: "household_survey_observer_cps",
            FORMS.tax_w2_1099.name: "tax_w2_observer",
            FORMS.wic.name: "wic_observer",
        }.get(form.name, f"{form.name}_observer")

        source = paths.SAMPLE_DATA_ROOT / form_file_name / f"{form_file_name}.parquet"
    if isinstance(source, str):
        source = Path(source)
    if isinstance(source, pd.DataFrame):
        data = source
    elif isinstance(source, Path):
        if source.suffix == ".hdf":
            data = pd.read_hdf(source)
        elif source.suffix == ".parquet":
            data = pd.read_parquet(source)
        else:
            raise ValueError(
                "Source path must either be a .hdf or a .parquet file. Provided "
                f"{source.suffix}"
            )
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"File located at {source} must contain a pandas DataFrame.")
    else:
        raise TypeError(
            f"Source {source} must be either a pandas DataFrame or a path to a "
            "file containing a pandas DataFrame."
        )
    return noise_form(form, data, configuration_tree, seed)


# TODO: add year as parameter to select the year of the decennial census to generate (MIC-3909)
# TODO: add default path: have the package install the small data in a known location and then
#  to make this parameter optional, with the default being the location of the small data that
#  is installed with the package (MIC-3884)
def generate_decennial_census(
    source: Union[Path, str, pd.DataFrame] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised decennial census data from un-noised data.

    :param source: A path to or pd.DataFrame of the un-noised source census data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised census data
    """
    return _generate_form(FORMS.census, source, seed, configuration)


def generate_american_communities_survey(
    source: Union[Path, str, pd.DataFrame] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised American Communities Survey (ACS) data from un-noised data.

    :param source: A path to or pd.DataFrame of the un-noised source ACS data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised ACS data
    """
    return _generate_form(FORMS.acs, source, seed, configuration)


def generate_current_population_survey(
    source: Union[Path, str, pd.DataFrame] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised Current Population Survey (CPS) data from un-noised data.

    :param source: A path to or pd.DataFrame of the un-noised source CPS data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised CPS data
    """
    return _generate_form(FORMS.cps, source, seed, configuration)


def generate_taxes_w2_and_1099(
    source: Union[Path, str, pd.DataFrame] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised W2 and 1099 data from un-noised data.

    :param source: A path to or pd.DataFrame of the un-noised source W2 and 1099 data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised W2 and 1099 data
    """
    return _generate_form(FORMS.tax_w2_1099, source, seed, configuration)


def generate_women_infants_and_children(
    source: Union[Path, str, pd.DataFrame] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised Women Infants and Children (WIC) data from un-noised data.

    :param source: A path to or pd.DataFrame of the un-noised source WIC data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised WIC data
    """
    return _generate_form(FORMS.wic, source, seed, configuration)


def generate_social_security(
    source: Union[Path, str, pd.DataFrame] = None,
    seed: int = 0,
    configuration: Union[Path, str, dict] = None,
) -> pd.DataFrame:
    """
    Generates noised Social Security (SSA) data from un-noised data.

    :param source: A path to or pd.DataFrame of the un-noised source SSA data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file or a dictionary to override the default configuration
    :return: A pd.DataFrame of noised SSA data
    """
    return _generate_form(FORMS.ssa, source, seed, configuration)
