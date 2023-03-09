from pathlib import Path

import pandas as pd
from vivarium.framework.configuration import ConfigTree, ConfigurationError
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.entities import Form


def get_randomness_stream(form: Form, seed: int) -> RandomnessStream:
    return RandomnessStream(form.value, lambda: pd.Timestamp("2020-04-01"), seed)


def get_default_configuration() -> ConfigTree:
    default_config_layers = [
        "base",
    ]
    noising_configuration = ConfigTree(layers=default_config_layers)
    yaml_path = Path("default_configuration.yaml")
    noising_configuration.update(yaml_path, layer="base")
    return noising_configuration
