from pathlib import Path

import pytest
import yaml
from vivarium.config_tree import ConfigTree

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


def test_get_configuration(user_configuration_yaml):
    config = get_configuration()
    assert config
    assert isinstance(config, ConfigTree)

    overridden_config = get_configuration(user_configuration_yaml)
    assert config != overridden_config
