from pathlib import Path
from typing import Union

import pandas as pd
import pytest

from pseudopeople.interface import generate_decennial_census


# TODO: possibly parametrize Forms?
def test_generate_decennial_census(
    dummy_census_data: Union[Path, str], dummy_config: Union[Path, str]
):
    data = pd.read_csv(dummy_census_data)
    noised_data = generate_decennial_census(
        path=dummy_census_data, seed=0, configuration=dummy_config
    )
    noised_data_same_seed = generate_decennial_census(
        path=dummy_census_data, seed=0, configuration=dummy_config
    )
    noised_data_different_seed = generate_decennial_census(
        path=dummy_census_data, seed=1, configuration=dummy_config
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
