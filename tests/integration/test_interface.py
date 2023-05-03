from typing import Callable

import numpy as np
import pandas as pd
import pytest

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants import metadata
from pseudopeople.interface import (
    _reformat_dates_for_noising,
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.schema_entities import COLUMNS, DATASETS, Dataset
from tests.integration.conftest import CELL_PROBABILITY, SEED

IDX_COLS = {
    DATASETS.census.name: [COLUMNS.simulant_id.name, COLUMNS.year.name],
    DATASETS.acs.name: [COLUMNS.simulant_id.name, COLUMNS.survey_date.name],
    DATASETS.cps.name: [COLUMNS.simulant_id.name, COLUMNS.survey_date.name],
    DATASETS.wic.name: [COLUMNS.simulant_id.name, COLUMNS.year.name],
    DATASETS.ssa.name: [COLUMNS.simulant_id.name, COLUMNS.ssa_event_type.name],
    DATASETS.tax_w2_1099.name: [
        COLUMNS.simulant_id.name,
        COLUMNS.tax_year.name,
        COLUMNS.employer_id.name,
    ],
    # DATASETS.tax_1040.name: "todo",
}

DATASET_FUNCS = {
    DATASETS.census.name: generate_decennial_census,
    DATASETS.acs.name: generate_american_community_survey,
    DATASETS.cps.name: generate_current_population_survey,
    DATASETS.ssa.name: generate_social_security,
    DATASETS.tax_w2_1099.name: generate_taxes_w2_and_1099,
    DATASETS.wic.name: generate_women_infants_and_children,
    # DATASETS.tax_1040.name: "todo",
}


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.ssa.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        "TODO: tax_1040",
    ],
)
def test_generate_dataset_from_sample_and_source(dataset_name: str, config, tmpdir, request):
    """Tests that the amount of noising is approximately the same whether we
    noise a single sample dataset or we concatenate and noise multiple datasets
    """
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)

    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    source = _generate_non_sample_data_root(dataset_name, tmpdir, data)
    noising_function = DATASET_FUNCS.get(dataset_name)
    noised_sample = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    noised_dataset = noising_function(seed=SEED, year=None, source=source, config=config)
    assert noised_dataset.shape == noised_sample.shape
    assert noised_dataset.columns.equals(noised_sample.columns)

    # Check that each columns level of noising are similar
    idx_cols = IDX_COLS.get(dataset_name)
    check_original = data.set_index(idx_cols)
    check_noised_sample = noised_sample.set_index(idx_cols)
    check_noised_dataset = noised_dataset.set_index(idx_cols)
    shared_idx_sample = pd.Index(
        set(check_noised_sample.index).intersection(set(check_original.index))
    )
    shared_idx_dataset = pd.Index(
        set(check_noised_dataset.index).intersection(set(check_original.index))
    )
    for col in check_noised_dataset.columns:
        original_missing_idx = check_original.index[check_original[col].isna()]
        both_missing_sample_idx = check_noised_sample.index[
            check_noised_sample[col].isna()
        ].intersection(original_missing_idx)
        both_missing_dataset_idx = check_noised_dataset.index[
            check_noised_dataset[col].isna()
        ].intersection(original_missing_idx)
        compare_sample_idx = shared_idx_sample.difference(both_missing_sample_idx)
        compare_dataset_idx = shared_idx_dataset.difference(both_missing_dataset_idx)
        noise_level_sample = (
            check_original.loc[compare_sample_idx, col].values
            != check_noised_sample.loc[compare_sample_idx, col].values
        ).mean()
        noise_level_dataset = (
            check_original.loc[compare_dataset_idx, col].values
            != check_noised_dataset.loc[compare_dataset_idx, col].values
        ).mean()
        np.isclose(noise_level_sample, noise_level_dataset, rtol=0.03)


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.ssa.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        "TODO: tax_1040",
    ],
)
def test_generate_dataset_and_col_noising(dataset_name: str, config, request):
    """Tests that noised datasets are generated and columns are noised as expected"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)

    noising_function = DATASET_FUNCS.get(dataset_name)
    dataset = DATASETS.get_dataset(dataset_name)

    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noised_data = noising_function(seed=SEED, year=None, config=config)

    _check_seed_behavior(noising_function, data, config, noised_data)

    # Check each column. We set the index for each dataset to be unique
    # identifiers b/c the original index gets reset after noising. Note that
    # the uniquely identifying columns must NOT be noised.
    # TODO: Replace this with the record ID column when implemented (MIC-4039)
    idx_cols = IDX_COLS.get(dataset_name)
    check_original = _reformat_dates_for_noising(data, dataset).set_index(idx_cols)
    check_noised = noised_data.set_index(idx_cols)
    # Ensure the idx_cols are unique
    assert check_original.index.duplicated().sum() == 0
    assert check_noised.index.duplicated().sum() == 0
    shared_idx = pd.Index(set(check_original.index).intersection(set(check_noised.index)))
    check_original = check_original.loc[shared_idx]
    check_noised = check_noised.loc[shared_idx]

    config = get_configuration(config)
    for col_name in check_noised.columns:
        col = COLUMNS.get_column(col_name)
        _check_dtype(col, noised_data)

        # Check that originally missing data remained missing
        originally_missing_idx = check_original.index[check_original[col.name].isna()]
        assert check_noised.loc[originally_missing_idx, col.name].isna().all()

        # Check for noising where applicable
        to_compare_idx = shared_idx.difference(originally_missing_idx)
        _check_column_noising(
            dataset_name,
            col,
            check_original,
            check_noised,
            config,
            to_compare_idx,
        )


def _generate_non_sample_data_root(data_dir_name, tmpdir, data):
    """Helper function to break the single sample dataset into two and save
    out to tmpdir to be used as a non-default 'source' argument
    """
    outdir = tmpdir.mkdir(data_dir_name)
    split_idx = int(len(data) / 2)
    data[:split_idx].to_parquet(outdir / f"{data_dir_name}_1.parquet")
    data[split_idx:].to_parquet(outdir / f"{data_dir_name}_2.parquet")
    return tmpdir


def _check_seed_behavior(noising_function, data, custom_config, noised_data):
    noised_data_same_seed = noising_function(seed=SEED, year=None, config=custom_config)
    noised_data_different_seed = noising_function(
        seed=SEED + 1, year=None, config=custom_config
    )

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)


def _check_dtype(col, noised_data):
    expected_dtype = col.dtype_name
    if expected_dtype == np.dtype(str):
        # str dtype is 'object'
        expected_dtype = np.dtype(object)
    assert noised_data[col.name].dtype == expected_dtype


def _check_column_noising(
    dataset_name, col, check_original, check_noised, config, to_compare_idx
):
    if col.noise_types:
        assert (
            check_original.loc[to_compare_idx, col.name].values
            != check_noised.loc[to_compare_idx, col.name].values
        ).any()

        noise_level = (
            check_original.loc[to_compare_idx, col.name].values
            != check_noised.loc[to_compare_idx, col.name].values
        ).mean()

        # Check that the amount of noising seems reasonable
        tmp_config = config[dataset_name][Keys.COLUMN_NOISE][col.name]
        includes_token_noising = [
            c for c in tmp_config if Keys.TOKEN_PROBABILITY in tmp_config[c].keys()
        ]
        # TODO [MIC-4052]: Come up with a more accurate values. There are token probabilities
        # and additional parameters to consider as well as the rtol when the
        # number of compared is small.
        expected_noise = 1 - (1 - CELL_PROBABILITY) ** len(col.noise_types)
        rtol = 0.5 if includes_token_noising else 0.11
        assert np.isclose(noise_level, expected_noise, rtol=rtol)
    else:  # No noising - should be identical
        assert (
            check_original.loc[to_compare_idx, col.name].values
            == check_noised.loc[to_compare_idx, col.name].values
        ).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.ssa.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        "TODO: tax_1040",
    ],
)
def test_generate_dataset_with_year(dataset_name: str, request):
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noised_data = request.getfixturevalue(f"noised_sample_data_2030_{dataset_name}")
    assert not data.equals(noised_data)


@pytest.mark.parametrize(
    "dataset_name, date_column",
    [
        (DATASETS.census.name, DATASETS.census.date_column),
        (DATASETS.tax_w2_1099.name, DATASETS.tax_w2_1099.date_column),
        (DATASETS.wic.name, DATASETS.wic.date_column),
        ("TODO: tax_1040", "todo"),
    ],
)
def test_dataset_filter_by_year(mocker, request, dataset_name: str, date_column: str):
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)

    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = request.getfixturevalue(f"noised_sample_data_2030_{dataset_name}")
    assert (noised_data[date_column] == 2030).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.ssa.name,
    ],
)
def test_dataset_filter_by_year_with_full_dates(mocker, request, dataset_name: str):
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = request.getfixturevalue(f"noised_sample_data_2030_{dataset_name}")
    dataset = DATASETS.get_dataset(dataset_name)
    dates = pd.DatetimeIndex(noised_data[dataset.date_column])
    if dataset == DATASETS.ssa:
        assert (dates.year <= 2030).all()
    else:
        assert (dates.year == 2030).all()


####################
# HELPER FUNCTIONS #
####################


def _mock_extract_columns(columns_to_keep, noised_dataset):
    return noised_dataset


def _mock_noise_dataset(
    dataset,
    dataset_data: pd.DataFrame,
    configuration,
    seed: int,
):
    """Mock noise_dataset that just returns unnoised data"""
    return dataset_data
