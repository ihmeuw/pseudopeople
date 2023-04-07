import sys
from pathlib import Path
from typing import Union

import pandas as pd

from pseudopeople.noise import noise_form
from pseudopeople.schema_entities import Form
from pseudopeople.utilities import get_configuration


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
    source: Union[Path, str, pd.DataFrame],
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
    return _generate_form(Form.CENSUS, source, seed, configuration)


def generate_american_communities_survey(
    source: Union[Path, str, pd.DataFrame],
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
    return _generate_form(Form.ACS, source, seed, configuration)


def generate_current_population_survey(
    source: Union[Path, str, pd.DataFrame],
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
    return _generate_form(Form.CPS, source, seed, configuration)


def generate_taxes_w2_and_1099(
    source: Union[Path, str, pd.DataFrame],
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
    return _generate_form(Form.TAX_W2_1099, source, seed, configuration)


def generate_women_infants_and_children(
    source: Union[Path, str, pd.DataFrame],
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
    return _generate_form(Form.WIC, source, seed, configuration)


def generate_social_security(
    source: Union[Path, str, pd.DataFrame],
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
    return _generate_form(Form.SSA, source, seed, configuration)


# Manual testing helper
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        my_path = Path(args[0])
        src = pd.read_csv(my_path, dtype=str, keep_default_na=False)
        out = generate_taxes_w2_and_1099(my_path)
        diff = src[
            ~src.astype(str).apply(tuple, 1).isin(out.astype(str).apply(tuple, 1))
        ]  # get all changed rows
        print(out.head())
        print(diff)
