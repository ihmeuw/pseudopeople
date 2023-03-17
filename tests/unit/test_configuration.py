from pathlib import Path

import pytest
from vivarium.config_tree import ConfigTree

from pseudopeople.entities import Form
from pseudopeople.utilities import (
    get_default_configuration,
    update_configuration_with_yaml,
)


@pytest.fixture
def user_configuration_yaml(tmp_path):
    text = """decennial_census:
    omission: 0.05
    first_name:
        nickname:
            row_noise_level: 0.05"""
    user_config_path = Path(f"{tmp_path}/test_configuration.yaml")
    with open(user_config_path, "w") as file:
        file.write(text)
    return user_config_path


def test_default_configuration():
    config = get_default_configuration()
    assert config
    assert isinstance(config, ConfigTree)
    # TODO: From Rajan: We should test that this configuration actually matches
    #  what we'd expect it to be. We can do this either by comparing it to the
    #  values in the yaml file, or by just confirming that the correct call to
    #  config_tree.update() was made in the function. The latter seems preferable
    #  to me as a unit test.


def test_user_configuration_file(user_configuration_yaml):
    """Test that a user config yaml will override the default configuration."""
    config = get_default_configuration()
    config = update_configuration_with_yaml(config, user_configuration_yaml)
    assert (
        config["decennial_census"]["omission"]
        == config["decennial_census"]["first_name"]["nickname"]["row_noise_level"]
    )
