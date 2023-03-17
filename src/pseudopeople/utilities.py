from pathlib import Path
from typing import Union

import pandas as pd
from vivarium.framework.configuration import ConfigTree, ConfigurationError
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.entities import Form


def get_randomness_stream(form: Form, seed: int) -> RandomnessStream:
    return RandomnessStream(form.value, lambda: pd.Timestamp("2020-04-01"), seed)


def get_default_configuration() -> ConfigTree:
    import pseudopeople

    default_config_layers = [
        "base",
        "user",
    ]
    noising_configuration = ConfigTree(layers=default_config_layers)
    BASE_DIR = Path(pseudopeople.__file__).resolve().parent
    yaml_path = BASE_DIR / "default_configuration.yaml"
    noising_configuration.update(yaml_path, layer="base")
    return noising_configuration


def update_configuration_with_yaml(
    configuration: ConfigTree, yaml_path: Union[Path, str]
) -> ConfigTree:
    """
    Updates a configuration ConfigTree with overrides from a provided YAML file.

    :param configuration: A ConfigTree configuration to override with a given YAML file
    :param yaml_path: A path to the YAML file defining overrides for configuration
    :return: a ConfigTree object updated with the configuration from the YAML
    """
    return configuration.update(yaml_path, layer="user")
