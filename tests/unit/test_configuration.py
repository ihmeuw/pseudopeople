import pytest
from vivarium.config_tree import ConfigTree

from pseudopeople.entities import Form
from pseudopeople.utilities import get_default_configuration


def test_default_configuration():
    config = get_default_configuration()
    assert config
    assert isinstance(config, ConfigTree)
    # TODO: From Rajan: We should test that this configuration actually matches
    # what we'd expect it to be. We can do this either by comparing it to the
    # values in the yaml file, or by just confirming that the correct call to
    # config_tree.update() was made in the function. The latter seems preferable
    # to me as a unit test.


@pytest.mark.skip(reason="TODO")
def test_user_configuration_file():
    pass
