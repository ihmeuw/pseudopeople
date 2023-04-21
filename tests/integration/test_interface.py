from typing import Callable

import numpy as np
import pandas as pd
import pytest

from pseudopeople.constants import metadata, paths
from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.schema_entities import COLUMNS, DATASETS

# TODO: Move into a metadata file and import metadata into prl
DATA_COLUMNS = ["year", "event_date", "survey_date", "tax_year"]


@pytest.mark.parametrize(
    "data_dir_name, noising_function, use_sample_data",
    [
        (DATASETS.census.name, generate_decennial_census, True),
        (DATASETS.census.name, generate_decennial_census, False),
        (DATASETS.acs.name, generate_american_community_survey, True),
        (DATASETS.acs.name, generate_american_community_survey, False),
        (DATASETS.cps.name, generate_current_population_survey, True),
        (DATASETS.cps.name, generate_current_population_survey, False),
        (DATASETS.ssa.name, generate_social_security, True),
        (DATASETS.ssa.name, generate_social_security, False),
        (DATASETS.tax_w2_1099.name, generate_taxes_w2_and_1099, True),
        (DATASETS.tax_w2_1099.name, generate_taxes_w2_and_1099, False),
        (DATASETS.wic.name, generate_women_infants_and_children, True),
        (DATASETS.wic.name, generate_women_infants_and_children, False),
        ("DATASETS.tax_1040.name", "todo", True),
        ("DATASETS.tax_1040.name", "todo", False),
    ],
)
def test_generate_dataset(
    data_dir_name: str, noising_function: Callable, use_sample_data: bool, tmpdir
):
    """Tests that noised datasets are generated and as expected. The 'use_sample_data'
    parameter determines whether to use the sample data (if True) or
    a non-default root directory with multiple datasets to compile (if False)
    """
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement dataset {data_dir_name}")

    sample_data_path = list(
        (paths.SAMPLE_DATA_ROOT / data_dir_name).glob(f"{data_dir_name}*")
    )[0]

    data = _load_data(sample_data_path)

    # Configure if default (sample data) is used or a different root directory
    if use_sample_data:
        source = None  # will default to using sample data
    else:
        source = _generate_non_default_data_root(
            data_dir_name, tmpdir, sample_data_path, data
        )

    noised_data = noising_function(seed=0, source=source)
    noised_data_same_seed = noising_function(seed=0, source=source)
    noised_data_different_seed = noising_function(seed=1, source=source)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)

    # Check each column's dtype
    for col in noised_data.columns:
        expected_dtype = [c.dtype_name for c in COLUMNS if c.name == col][0]
        if expected_dtype == np.dtype(str):
            # str dtype is 'object'
            expected_dtype = np.dtype(object)
        assert noised_data[col].dtype == expected_dtype


def _generate_non_default_data_root(data_dir_name, tmpdir, sample_data_path, data):
    """Helper function to break the single sample dataset into two and save
    out to tmpdir to be used as a non-default 'source' argument
    """
    outdir = tmpdir.mkdir(data_dir_name)
    suffix = sample_data_path.suffix
    split_idx = int(len(data) / 2)
    if suffix == ".parquet":
        data[:split_idx].to_parquet(outdir / f"{data_dir_name}_1{suffix}")
        data[split_idx:].to_parquet(outdir / f"{data_dir_name}_2{suffix}")
    elif suffix == ".hdf":
        data[:split_idx].to_hdf(
            outdir / f"{data_dir_name}_1{suffix}",
            "data",
            format="table",
            complib="bzip2",
            complevel=9,
            data_columns=DATA_COLUMNS,
        )
        data[split_idx:].to_hdf(
            outdir / f"{data_dir_name}_2{suffix}",
            "data",
            format="table",
            complib="bzip2",
            complevel=9,
            data_columns=DATA_COLUMNS,
        )
    else:
        raise NotImplementedError(f"Requires hdf or parquet, got {suffix}")
    return tmpdir


# TODO [MIC-4000]: add test that each col to get noised actually does get noised


@pytest.mark.parametrize(
    "data_dir_name, noising_function",
    [
        (DATASETS.census.name, generate_decennial_census),
        (DATASETS.acs.name, generate_american_community_survey),
        (DATASETS.cps.name, generate_current_population_survey),
        (DATASETS.ssa.name, generate_social_security),
        (DATASETS.tax_w2_1099.name, generate_taxes_w2_and_1099),
        (DATASETS.wic.name, generate_women_infants_and_children),
        ("DATASETS.tax_1040.name", "todo"),
    ],
)
def test_generate_dataset_with_year(data_dir_name: str, noising_function: Callable):
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement dataset {data_dir_name}")
    data_path = paths.SAMPLE_DATA_ROOT / data_dir_name / f"{data_dir_name}.parquet"
    data = _load_data(data_path)

    noised_data = noising_function(year=2020, seed=0)
    noised_data_same_seed = noising_function(year=2020, seed=0)
    noised_data_different_seed = noising_function(year=2020, seed=1)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)


def _mock_extract_columns(columns_to_keep, noised_dataset):
    return noised_dataset


@pytest.mark.parametrize(
    "data_dir_name, noising_function, date_column",
    [
        (DATASETS.census.name, generate_decennial_census, DATASETS.census.date_column),
        (
            DATASETS.tax_w2_1099.name,
            generate_taxes_w2_and_1099,
            DATASETS.tax_w2_1099.date_column,
        ),
        (DATASETS.wic, generate_women_infants_and_children, DATASETS.wic.date_column),
        (metadata.DatasetNames.TAXES_1040, "todo", "todo"),
    ],
)
def test_dataset_filter_by_year(
    mocker, data_dir_name: str, noising_function: Callable, date_column: str
):
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement dataset {data_dir_name}")

    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = noising_function(year=2020)

    assert (noised_data[date_column] == 2020).all()


def _mock_noise_dataset(
    dataset,
    dataset_data: pd.DataFrame,
    configuration,
    seed: int,
):
    """Mock noise_dataset that just returns unnoised data"""
    return dataset_data


@pytest.mark.parametrize(
    "data_dir_name, noising_function, dataset",
    [
        (metadata.DatasetNames.ACS, generate_american_community_survey, DATASETS.acs),
        (metadata.DatasetNames.CPS, generate_current_population_survey, DATASETS.cps),
        (metadata.DatasetNames.SSA, generate_social_security, DATASETS.ssa),
    ],
)
def test_dataset_filter_by_year_with_full_dates(
    mocker, data_dir_name: str, noising_function: Callable, dataset: DATASETS
):
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = noising_function(year=2020)

    dates = pd.DatetimeIndex(noised_data[dataset.date_column])
    if dataset == DATASETS.ssa:
        assert (dates.year <= 2020).all()
    else:
        assert (dates.year == 2020).all()


def _load_data(data_path):
    if data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)
    elif data_path.suffix == ".hdf":
        data = pd.read_hdf(data_path)
    else:
        raise NotImplementedError(f"Expected hdf or parquet but got {data_path.suffix}")

    return data
