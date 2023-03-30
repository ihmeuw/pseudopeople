import sys
from pathlib import Path
from typing import Union

import pandas as pd

from pseudopeople.noise import noise_form
from pseudopeople.schema_entities import Form
from pseudopeople.utilities import get_configuration


# TODO: add year as parameter to select the year of the decennial census to generate (MIC-3909)
# TODO: add default path: have the package install the small data in a known location and then
#  to make this parameter optional, with the default being the location of the small data that
#  is installed with the package (MIC-3884)
def generate_decennial_census(
    path: Union[Path, str], seed: int = 0, configuration: Union[Path, str] = None
):
    """
    Generates a noised decennial census data from un-noised data.

    :param path: A path to the un-noised source census data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file to modify default values
    :return: A pd.DataFrame of noised census data
    """
    configuration_tree = get_configuration(configuration)
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    return noise_form(Form.CENSUS, data, configuration_tree, seed)


def generate_w2(
    path: Union[Path, str], seed: int = 0, configuration: Union[Path, str] = None
):
    """
    Generates a noised W2 data from un-noised data.

    :param path: A path to the un-noised source W2 data
    :param seed: An integer seed for randomness
    :param configuration: (optional) A path to a configuration YAML file to modify default values
    :return: A pd.DataFrame of noised W2 data
    """
    configuration_tree = get_configuration(configuration)
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    return noise_form(Form.TAX_W2_1099, data, configuration_tree, seed)


# Manual testing helper
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        my_path = Path(args[0])
        src = pd.read_csv(my_path, dtype=str, keep_default_na=False)
        out = generate_w2(my_path)
        diff = src[
            ~src.astype(str).apply(tuple, 1).isin(out.astype(str).apply(tuple, 1))
        ]  # get all changed rows
        print(out.head())
        print(diff)
