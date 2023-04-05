from pathlib import Path

import pandas as pd
import pytest
import yaml
from vivarium.config_tree import ConfigTree
from vivarium.framework.randomness import RandomnessStream

import pseudopeople
from pseudopeople.utilities import get_configuration

RANDOMNESS0 = RandomnessStream(
    key="test_column_noise", clock=lambda: pd.Timestamp("2020-09-01"), seed=0
)


@pytest.fixture
def user_configuration_yaml(tmp_path):
    user_config_path = Path(f"{tmp_path}/test_configuration.yaml")
    config = {
        "decennial_census": {
            "omission": 0.05,
            "first_name": {"nickname": {"row_noise_level": 0.05}},
        }
    }
    with open(user_config_path, "w") as file:
        yaml.dump(config, file)
    return user_config_path


def test_get_configuration(mocker):
    """Tests that the default configuration can be retrieved."""
    mock = mocker.patch("pseudopeople.utilities.ConfigTree")
    _ = get_configuration()
    mock.assert_called_once_with(
        data=Path(pseudopeople.__file__).resolve().parent / "default_configuration.yaml",
        layers=["base", "user"],
    )


def test_get_configuration_with_user_override(user_configuration_yaml, mocker):
    """Tests that the default configuration get updated when a user configuration is supplied."""
    mock = mocker.patch("pseudopeople.utilities.ConfigTree")
    _ = get_configuration(user_configuration_yaml)
    mock.assert_called_once_with(
        data=Path(pseudopeople.__file__).resolve().parent / "default_configuration.yaml",
        layers=["base", "user"],
    )
    update_calls = [
        call
        for call in mock.mock_calls
        if ".update({" in str(call) and "layer='user'" in str(call)
    ]
    assert len(update_calls) == 1


def test_validate_miswrite_ages_fails_if_includes_0():
    """Test that a runtime error is thrown if the user includes 0 as a possible perturbation"""
    perturbations = [-1, 0, 1]
    with pytest.raises(ValueError, match="Cannot include 0"):
        get_configuration(
            {
                "decennial_census": {
                    "age": {
                        "age_miswriting": {
                            "row_noise_level": 1,
                            "possible_perturbations": perturbations,
                        },
                    },
                },
            },
        )


def test_validate_miswrite_ages_if_probabilities_do_not_add_to_1():
    """Test that runtimerrors if probs do not add up to 1"""
    perturbations = {-1: 0.1, 1: 0.8}  # does not sum to 1

    with pytest.raises(ValueError, match="must sum to 1"):
        get_configuration(
            {
                "decennial_census": {
                    "age": {
                        "age_miswriting": {
                            "possible_perturbations": perturbations,
                        },
                    },
                },
            },
        )


@pytest.mark.parametrize("user_config_type", ["dict", "path"])
def test_format_miswrite_ages(user_config_type, tmp_path):
    """Test that user-supplied dictionary properly updates ConfigTree object.
    This includes zero-ing out default values that don't exist in the user config
    """
    user_config = {
        "decennial_census": {
            "age": {
                "age_miswriting": {
                    "possible_perturbations": [-2, -1, 2],
                },
            },
        },
    }
    if user_config_type == "path":
        filepath = tmp_path / "user_dict.yaml"
        with open(filepath, "w") as file:
            yaml.dump(user_config, file)
        user_config = filepath

    new_dict = get_configuration(user_config).decennial_census.age.age_miswriting.to_dict()
    default_dict = get_configuration().decennial_census.age.age_miswriting.to_dict()
    assert default_dict["row_noise_level"] == new_dict["row_noise_level"]
    assert default_dict["token_noise_level"] == new_dict["token_noise_level"]
    # check that 1 got replaced with 0 probability
    assert new_dict["possible_perturbations"][1] == 0
    # check that others have 1/3 probability
    for p in [-2, -1, 2]:
        assert new_dict["possible_perturbations"][p] == 1 / 3
