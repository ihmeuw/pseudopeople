import os
from pathlib import Path

import numpy as np
import psutil
import pytest
from _pytest.tmpdir import TempPathFactory
from dask.distributed import LocalCluster, get_client
from dask.system import CPU_COUNT
from packaging.version import parse
from pytest_mock import MockerFixture

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.interface import (
    _get_data_changelog_version,
    set_up_dask_client,
    validate_source_compatibility,
)
from pseudopeople.schema_entities import DATASET_SCHEMAS
from tests.utilities import is_on_slurm

CENSUS = DATASET_SCHEMAS.get_dataset_schema(DatasetNames.CENSUS)


# TODO [MIC-4546]: stop hardcoding the data version number
@pytest.fixture(scope="module")
def simulated_data_changelog_path(tmp_path_factory: TempPathFactory) -> Path:
    """Returns the path to where simualted data would live alongside their
    respective CHANGELOG.rst. This fixture creates the changelog but does
    not actually contain any simulated data.
    """
    tmp_path = str(tmp_path_factory.getbasetemp())
    filepath = f"{tmp_path}/CHANGELOG.rst"
    changelog_content = (
        "**1.4.2 - some other date**"
        "\n\n"
        " - Did some other things"
        "\n\n"
        "**1.4.1 - some date**"
        "\n\n"
        " - Did some things"
        "\n"
    )
    with open(filepath, "w") as file:
        file.write(changelog_content)

    # Make fake data directory for census
    tmp_path_factory.mktemp(CENSUS.name, numbered=False)

    return Path(tmp_path)


def test__get_data_changelog_version(simulated_data_changelog_path: Path) -> None:
    """Test that the data version is extracted from the CHANGELOG correctly"""
    assert _get_data_changelog_version(
        simulated_data_changelog_path / "CHANGELOG.rst"
    ) == parse("1.4.2")


def mock_data_version(version: str, mocker: MockerFixture) -> None:
    mocker.patch(
        "pseudopeople.interface._get_data_changelog_version", return_value=parse(version)
    )


def test_validate_source_compatibility_passes(simulated_data_changelog_path: Path) -> None:
    """Baseline test for validate_source_compatibility function"""
    validate_source_compatibility(simulated_data_changelog_path, CENSUS)


def test_validate_source_compatibility_no_changelog_error(tmp_path: Path) -> None:
    no_changelog_dir = tmp_path / "no_changelog"
    # No changelog is made
    (no_changelog_dir / CENSUS.name).mkdir(parents=True)
    with pytest.raises(
        DataSourceError,
        match="An older version of simulated population data has been provided.",
    ):
        validate_source_compatibility(no_changelog_dir, CENSUS)


@pytest.mark.parametrize(
    "version, match",
    [
        ("1.4.1", "The simulated population data has been corrupted."),
        ("1.4.3", "A newer version of simulated population data has been provided."),
        ("1.4.12", "A newer version of simulated population data has been provided."),
    ],
)
def test_validate_source_compatibility_bad_version_errors(
    version: str, match: str, simulated_data_changelog_path: Path, mocker: MockerFixture
) -> None:
    mock_data_version(version, mocker)
    with pytest.raises(DataSourceError, match=match):
        validate_source_compatibility(simulated_data_changelog_path, CENSUS)


def test_validate_source_compatibility_wrong_directory(tmp_path: Path) -> None:
    bad_path = tmp_path / "wrong_directory"
    bad_path.mkdir()
    with pytest.raises(FileNotFoundError, match="Could not find 'decennial_census' in"):
        validate_source_compatibility(bad_path, CENSUS)


def test_set_up_dask_client_default() -> None:

    # There should be no dask client yet
    with pytest.raises(ValueError):
        client = get_client()

    set_up_dask_client()
    client = get_client()
    workers = client.scheduler_info()["workers"]
    assert len(workers) == CPU_COUNT
    assert all(worker["nthreads"] == 1 for worker in workers.values())
    if is_on_slurm():
        try:
            available_memory = float(os.environ["SLURM_MEM_PER_NODE"]) / 1024
        except KeyError:
            raise RuntimeError(
                "You are on Slurm but SLURM_MEM_PER_NODE is not set. "
                "It is likely that you are SSHed onto a node (perhaps using VSCode). "
                "In this case, dask will assign the total memory of the node to each "
                "worker instead of the allocated memory from the srun call. "
                "Pseudopeople should only be used on Slurm directly on the node "
                "assigned via an srun (both for pytests as well as actual work)."
            )
    else:
        available_memory = psutil.virtual_memory().total / (1024 ** 3)
    assert np.isclose(sum(worker["memory_limit"] / 1024**3 for worker in workers.values()), available_memory, rtol=0.01)


def test_set_up_dask_client_custom() -> None:
    memory_limit = 1  # gb
    n_workers = 3
    cluster = LocalCluster(
        name="custom",
        n_workers=n_workers,
        threads_per_worker=2,
        memory_limit=memory_limit * 1024**3,
    )
    client = cluster.get_client()
    set_up_dask_client()
    client = get_client()
    assert client.cluster.name == "custom"
    workers = client.scheduler_info()["workers"]
    assert len(workers) == 3
    assert all(worker["nthreads"] == 2 for worker in workers.values())
    assert sum(worker["memory_limit"] / 1024**3 for worker in workers.values()) == memory_limit * n_workers
