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
    # Shut down a client if it exists
    try:
        client = get_client()
        client.shutdown()  # type: ignore[no-untyped-call]
    except ValueError:
        pass
    finally:
        # There should be no dask client at this point
        with pytest.raises(ValueError):
            client = get_client()

    set_up_dask_client()

    if is_on_slurm():
        try:
            available_memory = float(os.environ["SLURM_MEM_PER_NODE"]) / 1024
        except KeyError:
            raise RuntimeError(
                "NOTE: This RuntimeError is expected if you are using VSCode on the cluster!\n\n"
                "You are on Slurm but SLURM_MEM_PER_NODE is not set; it is likely "
                "that you are SSHed onto a node (perhaps using VSCode?). "
                "In this case, dask will assign the total memory of the node to the "
                "cluster instead of the allocated memory from the srun call. "
                "Pseudopeople should only be used on Slurm directly on the node "
                "assigned via an srun (both for pytests as well as actual work)."
            )
    else:
        available_memory = psutil.virtual_memory().total / (1024**3)

    _check_cluster_attrs(
        cluster_name="pseudopeople_dask_cluster",
        memory_limit=available_memory,
        n_workers=CPU_COUNT,
        threads_per_worker=1,
    )


def test_set_up_dask_client_existing_cluster() -> None:
    cluster_name = "custom"
    memory_limit = 1  # gb
    n_workers = 3
    threads_per_worker = 2

    # Manually create a cluster
    cluster = LocalCluster(  # type: ignore[no-untyped-call]
        name=cluster_name,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit * 1024**3,
    )
    cluster.get_client()  # type: ignore[no-untyped-call]
    _check_cluster_attrs(
        cluster_name=cluster_name,
        memory_limit=memory_limit * n_workers,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker
    )
    
    # Call the dask client setup function
    set_up_dask_client()
    
    # Make sure that the cluster hasn't been changed
    assert get_client().cluster == cluster
    _check_cluster_attrs(
        cluster_name=cluster_name,
        memory_limit=memory_limit * n_workers,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker
    )

####################
# Helper Functions #
####################

def _check_cluster_attrs(cluster_name: str, memory_limit: int | float, n_workers: int, threads_per_worker: int) -> None:
    cluster = get_client().cluster
    assert isinstance(cluster, LocalCluster)
    assert cluster.name == cluster_name
    workers = cluster.scheduler_info["workers"]
    assert len(workers) == n_workers
    assert all(worker["nthreads"] == threads_per_worker for worker in workers.values())
    assert np.isclose(
        sum(worker["memory_limit"] / 1024**3 for worker in workers.values()),
        memory_limit,
        rtol=0.01,
    )
