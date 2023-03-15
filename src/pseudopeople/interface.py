import sys
from pathlib import Path
from typing import Union

import pandas as pd

from pseudopeople.entities import Form
from pseudopeople.noise import noise_form
from pseudopeople.utilities import get_default_configuration


def generate_decennial_census(path: Union[Path, str]):
    """
    Generates a noised decennial census data from un-noised data.

    :param path: A path to the un-noised source census data
    :return: A pd.DataFrame of noised census data
    """
    configuration = get_default_configuration()
    data = pd.read_csv(path)
    return noise_form(Form.CENSUS, data, configuration, seed=0)  # XXX: what is seed here?


# Manual testing helper
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        my_path = Path(args[0])
        src = pd.read_csv(my_path)
        out = generate_decennial_census(my_path)
        diff = src[
            ~src.astype(str).apply(tuple, 1).isin(out.astype(str).apply(tuple, 1))
        ]  # get all changed rows
        print(out.head())
        print(diff)
