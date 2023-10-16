from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--limit",
        action="store",
        default=-1,
        type=int,
        help="Maximum number of parameterized tests to run",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        # Automatically tag all tests in the tests/integration dir as slow
        if Path(item.parent.path).parent.stem == "integration":
            item.add_marker(pytest.mark.slow)
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

    # Limit the number of permutations of parametrised tests to run.
    limit = config.getoption("--limit")
    if limit > 0:
        tests_by_name = {item.name: item for item in items}
        # Add the name of parametrized base tests to this list.
        tests_to_skip_parametrize = ["test_noise_order"]

        for base_name in tests_to_skip_parametrize:
            to_skip = [t for n, t in tests_by_name.items() if base_name in n][limit:]
            for t in to_skip:
                t.add_marker("skip")


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)
