import pytest
from vivarium.config_tree import ConfigTree

from pseudopeople.entities import Form
from pseudopeople.utilities import get_default_configuration


def test_default_configuration():
    config = get_default_configuration()
    assert config
    assert isinstance(config, ConfigTree)


@pytest.mark.skip(reason="TODO")
def test_user_configuration_file():
    pass
