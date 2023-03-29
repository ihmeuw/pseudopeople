import pytest
import yaml

from pseudopeople.utilities import get_configuration


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def user_config_path(tmp_path_factory):
    """This simply copies the default config file to a temp directory
    to be used as a user-provided config file in integration tests
    """
    config = get_configuration().to_dict()  # gets default config
    config_path = tmp_path_factory.getbasetemp() / "dummy_config.yaml"
    with open(config_path, "w") as file:
        yaml.dump(config, file)

    return config_path
