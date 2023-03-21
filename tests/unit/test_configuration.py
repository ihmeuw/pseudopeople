from pathlib import Path

import pytest
import yaml

import pseudopeople
from pseudopeople.utilities import get_configuration


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
        if "update" in str(call)
        and "user" in str(call)
        and str(user_configuration_yaml) in str(call)
    ]
    assert len(update_calls) == 1
