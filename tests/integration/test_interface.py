from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytest
import yaml
from vivarium.config_tree import ConfigTree

from pseudopeople.interface import generate_decennial_census
from pseudopeople.utilities import get_configuration, get_possible_indices_to_noise


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


def test_noise_decennial_census_with_two_noise_functions(dummy_census_data, tmp_path_factory):
    # todo: Make config tree with 2 function calls
    # Make simple config tree to test 2 noise functions work together
    config_tree = ConfigTree(
        {
            "decennial_census": {
                "first_name": {
                    "missing_data": {"row_noise_level": 0.01},
                },
                "state": {
                    "missing_data": {"row_noise_level": 0.01},
                    "incorrect_select": {"row_noise_level": 0.01},
                },
                "race_ethnicity": {
                    "missing_data": {"row_noise_level": 0.01},
                    "incorrect_select": {"row_noise_level": 0.01},
                },
                "duplication": 0.01,
                "omission": 0.01,
            }
        }
    )
    config_dict = config_tree.to_dict()
    config_path = tmp_path_factory.getbasetemp() / "test_multiple_ooise_config.yaml"
    with open(config_path, "w") as file:
        yaml.dump(config_dict, file)

    data = pd.read_csv(dummy_census_data)
    noised_data = generate_decennial_census(
        path=dummy_census_data, seed=0, configuration=config_path
    )

    config = config_tree["decennial_census"]
    for col in noised_data:
        if col in config:
            non_missing_idx = get_possible_indices_to_noise(data[col])
            # todo: Use when np.NaN from post-processing have been handled
            # Check rows with missing values did not change - change NaNs to empty string?
            # assert (data.loc[
            #             data.index.difference(non_missing_idx), col] == noised_data.loc[
            #     noised_data.index.difference(non_missing_idx), col]).all()
            old = data.loc[non_missing_idx, col]
            noised_col = noised_data.loc[non_missing_idx, col]
            assert len(old) == len(noised_col)
            actual_noise_rate = (noised_col != old).sum() / len(noised_col)
            noise_types = [k for k in config[col]]
            noise_rates = [
                config[col][noise_type]["row_noise_level"] for noise_type in noise_types
            ]
            expected_noise_rate = 1 - np.prod([1 - x for x in noise_rates])
            assert np.isclose(actual_noise_rate, expected_noise_rate, rtol=0.10)


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
