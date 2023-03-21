from pathlib import Path
from typing import Union

import pandas as pd
from vivarium.framework.configuration import ConfigTree
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.entities import Form


def get_randomness_stream(form: Form, seed: int) -> RandomnessStream:
    return RandomnessStream(form.value, lambda: pd.Timestamp("2020-04-01"), seed)


def get_configuration(user_yaml_path: Union[Path, str] = None) -> ConfigTree:
    """
    Gets a noising configuration ConfigTree, optionally overridden by a user-provided YAML.

    :param user_yaml_path: A path to the YAML file defining user overrides for the defaults
    :return: a ConfigTree object of the noising configuration
    """
    import pseudopeople

    default_config_layers = [
        "base",
        "user",
    ]
    noising_configuration = ConfigTree(
        data=Path(pseudopeople.__file__).resolve().parent / "default_configuration.yaml",
        layers=default_config_layers,
    )
    if user_yaml_path:
        noising_configuration.update(user_yaml_path, layer="user")
    return noising_configuration
