from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytest

from pseudopeople.interface import generate_decennial_census
from pseudopeople.utilities import get_configuration


# TODO: possibly parametrize Forms?
def test_generate_decennial_census(
    decennial_census_data_path: Union[Path, str], user_config_path: Union[Path, str]
):
    data = pd.read_csv(decennial_census_data_path)
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
    # TODO: Confirm correct columns exist once the interface functions
    # modify them
    # TODO: if we sort out dtype schemas
    # for col in noised_data.columns:
    # assert data[col].dtype == noised_data[col].dtype
    # TODO: Iterate through cols and check that the percentage of errors makes sense
    # eg, if 25% typographic error and 1% OCR
    # 1. Use a default config file
    # 2.

    config = get_configuration(user_config_path)["decennial_census"]

    # Confirm omission and duplication seems reasonable
    # TODO: when omission function gets implemented.
    orig_idx = data.index
    noised_idx = noised_data.index
    # assert np.isclose(len(set(orig_idx) - set(noised_idx)) / len(data), config.omission)
    # TODO: when duplication function gets implemented
    # assert np.isclose(noised_data.duplicated().sum() / len(data), config.duplication)

    # Check that column-level noise seem reasonable
    # NOTE: this is not perfect because (1) it is only looking at row-level
    # noise and not token-based noise and (2) it is not accounting for the
    # fact that noising can occur on duplicated rows which have been removed
    # for comparison purposes.
    common_idx = set(orig_idx).intersection(set(noised_idx))
    common_data = data.loc[common_idx]
    common_noised_data = noised_data.loc[common_idx].drop_duplicates()
    assert common_data.shape == common_noised_data.shape
    for col in noised_data:
        if col in config:
            actual_noise_rate = (common_data[col] != common_noised_data[col]).mean()
            noise_types = [k for k in config[col]]
            noise_rates = [
                config[col][noise_type]["row_noise_level"] for noise_type in noise_types
            ]
            expected_noise_rate = 1 - np.prod([1 - x for x in noise_rates])
            assert np.isclose(actual_noise_rate, expected_noise_rate, rtol=0.07)
        else:
            assert (common_data[col] == common_noised_data[col]).all()


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
