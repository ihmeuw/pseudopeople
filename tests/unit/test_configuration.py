from pathlib import Path

import pytest
import yaml
from vivarium.config_tree import ConfigTree

from pseudopeople.configuration import DEFAULT_NOISE_VALUES, get_configuration
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import FORMS


@pytest.fixture
def user_configuration_yaml(tmp_path):
    user_config_path = Path(f"{tmp_path}/test_configuration.yaml")
    config = {
        "decennial_census": {
            "row_noise": {"omission": {"probability": 0.05}},
            "column_noise": {"first_name": {"nickname": {"row_noise_level": 0.05}}},
        }
    }
    with open(user_config_path, "w") as file:
        yaml.dump(config, file)
    return user_config_path


def test_get_default_configuration(mocker):
    """Tests that the default configuration can be retrieved."""
    mock = mocker.patch("pseudopeople.configuration.ConfigTree")
    _ = get_configuration()
    mock.assert_called_once_with(layers=["baseline", "default", "user"])


def test_default_configuration_structure():
    """Test that the default configuration structure is correct"""
    config = get_configuration()
    # Check forms
    assert set(f.name for f in FORMS) == set(config.keys())
    for form in FORMS:
        # Check row noise
        for row_noise in form.row_noise_types:
            config_probability = config[form.name].row_noise[row_noise.name].probability
            default_probability = (
                DEFAULT_NOISE_VALUES.get(form.name, {})
                .get("row_noise", {})
                .get(row_noise.name, {})
                .get("probability", "no default")
            )
            if default_probability == "no default":
                assert config_probability == getattr(NOISE_TYPES, row_noise.name).probability
            else:
                assert config_probability == default_probability
        for col in form.columns:
            for noise_type in col.noise_types:
                config_level = config[form.name].column_noise[col.name][noise_type.name]
                baseline_level = getattr(NOISE_TYPES, noise_type.name)
                # FIXME: Is there a way to allow for adding new keys when they
                # don't exist in baseline? eg the for if loops below depend on their
                # being row_noise, token_noise, and additional parameters at the
                # baseline level ('noise_type in col.noise_types')
                # Would we ever want to allow for adding non-baseline default noise?
                if noise_type.row_noise_level:
                    config_row_noise_level = config_level.row_noise_level
                    default_row_noise_level = (
                        DEFAULT_NOISE_VALUES.get(form.name, {})
                        .get("column_noise", {})
                        .get(col.name, {})
                        .get(noise_type.name, {})
                        .get("row_noise_level", "no default")
                    )
                    if default_row_noise_level == "no default":
                        assert config_row_noise_level == baseline_level.row_noise_level
                    else:
                        assert config_row_noise_level == default_row_noise_level
                if noise_type.token_noise_level:
                    config_token_noise_level = config_level.token_noise_level
                    default_token_noise_level = (
                        DEFAULT_NOISE_VALUES.get(form.name, {})
                        .get("column_noise", {})
                        .get(col.name, {})
                        .get(noise_type.name, {})
                        .get("token_noise_level", "no default")
                    )
                    if default_token_noise_level == "no default":
                        assert config_token_noise_level == baseline_level.token_noise_level
                    else:
                        assert config_token_noise_level == default_token_noise_level
                if noise_type.additional_parameters:
                    config_additional_parameters = {
                        k: v
                        for k, v in config_level.to_dict().items()
                        if k not in ["row_noise_level", "token_noise_level"]
                    }
                    default_additional_parameters = (
                        DEFAULT_NOISE_VALUES.get(form.name, {})
                        .get("column_noise", {})
                        .get(col.name, {})
                        .get(noise_type.name, {})
                    )
                    default_additional_parameters = {
                        k: v
                        for k, v in default_additional_parameters.items()
                        if k not in ["row_noise_level", "token_noise_level"]
                    }
                    if default_additional_parameters == {}:
                        assert (
                            config_additional_parameters
                            == baseline_level.additional_parameters
                        )
                    else:
                        # Confirm config includes default values
                        for key, value in default_additional_parameters.items():
                            assert config_additional_parameters[key] == value
                        # Check that non-default values are baseline
                        baseline_keys = [
                            k
                            for k in config_additional_parameters
                            if k not in default_additional_parameters
                        ]
                        for key, value in config_additional_parameters.items():
                            if key not in baseline_keys:
                                continue
                            assert baseline_level.additional_parameters[key] == value


def test_get_configuration_with_user_override(user_configuration_yaml, mocker):
    """Tests that the default configuration get updated when a user configuration is supplied."""
    mock = mocker.patch("pseudopeople.configuration.ConfigTree")
    _ = get_configuration(user_configuration_yaml)
    mock.assert_called_once_with(layers=["baseline", "default", "user"])
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
                    "column_noise": {
                        "age": {
                            "age_miswriting": {
                                "row_noise_level": 1,
                                "possible_perturbations": perturbations,
                            },
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
                    "column_noise": {
                        "age": {
                            "age_miswriting": {
                                "possible_perturbations": perturbations,
                            },
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
            "column_noise": {
                "age": {
                    "age_miswriting": {
                        "possible_perturbations": [-2, -1, 2],
                    },
                },
            },
        },
    }
    if user_config_type == "path":
        filepath = tmp_path / "user_dict.yaml"
        with open(filepath, "w") as file:
            yaml.dump(user_config, file)
        user_config = filepath

    new_dict = get_configuration(
        user_config
    ).decennial_census.column_noise.age.age_miswriting.to_dict()
    default_dict = (
        get_configuration().decennial_census.column_noise.age.age_miswriting.to_dict()
    )
    assert default_dict["row_noise_level"] == new_dict["row_noise_level"]
    # check that 1 got replaced with 0 probability
    assert new_dict["possible_perturbations"][1] == 0
    # check that others have 1/3 probability
    for p in [-2, -1, 2]:
        assert new_dict["possible_perturbations"][p] == 1 / 3
