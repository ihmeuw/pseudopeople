from pathlib import Path

import pytest
import yaml
from packaging.version import parse

from pseudopeople.exceptions import DataSourceError
from pseudopeople.interface import _get_data_version, validate_source_compatibility


# TODO [MIC-4546]: stop hardcoding the data version number
@pytest.fixture()
def simulated_data_metadata_path(tmpdir_factory):
    """Returns the path to where simualted data would live alongside their
    respective CHANGELOG.rst. This fixture creates the changelog but does
    not actually contain any simulated data.
    """
    tmp_path = str(tmpdir_factory.getbasetemp())
    filepath = f"{tmp_path}/metadata.yaml"
    metadata_content = {"data_version": "2.0.0"}
    with open(filepath, "w") as file:
        yaml.dump(metadata_content, file)

    # Create a dir with no changelog
    tmpdir_factory.mktemp("no_metadata")

    return Path(tmp_path)


def test__get_data_changelog_version(simulated_data_metadata_path):
    """Test that the data version is extracted from the CHANGELOG correctly"""
    assert _get_data_version(simulated_data_metadata_path / "metadata.yaml") == parse("2.0.0")


def mock_data_version(version, mocker):
    mocker.patch("pseudopeople.interface._get_data_version", return_value=parse(version))


def test_validate_source_compatibility_passes(simulated_data_metadata_path):
    """Baseline test for validate_source_compatibility function"""
    validate_source_compatibility(simulated_data_metadata_path)


def test_validate_source_compatibility_no_metadata_error(simulated_data_metadata_path):
    with pytest.raises(
        DataSourceError,
        match="An older version of simulated population data has been provided.",
    ):
        validate_source_compatibility(simulated_data_metadata_path / "no_metadata")


@pytest.mark.parametrize(
    "version, match",
    [
        ("1.4.1", "The simulated population data has been corrupted."),
        ("2.1.0", "A newer version of simulated population data has been provided."),
        ("2.4.12", "A newer version of simulated population data has been provided."),
    ],
)
def test_validate_source_compatibility_bad_version_errors(
    version, match, simulated_data_metadata_path, mocker
):
    mock_data_version(version, mocker)
    with pytest.raises(DataSourceError, match=match):
        validate_source_compatibility(simulated_data_metadata_path)
