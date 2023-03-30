from pathlib import Path
from typing import Union

import pandas as pd
import pytest

from pseudopeople.interface import generate_decennial_census


# TODO: possibly parametrize Forms?
def test_generate_decennial_census(
    decennial_census_data_path: Union[Path, str], user_config_path: Union[Path, str]
):
    data = pd.read_csv(decennial_census_data_path, dtype=str, keep_default_na=False)

    # TODO: Refactor this check into a separate test
    noised_data = generate_decennial_census(
        path=decennial_census_data_path, seed=0, configuration=user_config_path
    )
    noised_data_same_seed = generate_decennial_census(
        path=decennial_census_data_path, seed=0, configuration=user_config_path
    )
    noised_data_different_seed = generate_decennial_census(
        path=decennial_census_data_path, seed=1, configuration=user_config_path
    )

    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)
    assert not data.equals(noised_data)
    assert set(noised_data.columns) == set(data.columns)


@pytest.mark.skip(reason="TODO")
def test_generate_acs():
    pass


@pytest.mark.skip(reason="TODO")
def test_generate_cps():
    pass


@pytest.mark.skip(reason="TODO")
def test_generate_wic():
    pass


@pytest.mark.skip(reason="TODO")
def test_generate_ssa():
    pass


@pytest.mark.skip(reason="TODO")
def test_generate_tax_w2_1099():
    pass


@pytest.mark.skip(reason="TODO")
def test_generate_tax_1040():
    pass
