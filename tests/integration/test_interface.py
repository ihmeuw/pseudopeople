import numpy as np
import pandas as pd
import pytest

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.interface import (
    _reformat_dates_for_noising,
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.schema_entities import COLUMNS, DATASETS
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

DATASET_GENERATION_FUNCS = {
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
    noised_sample = request.getfixturevalue(f"noised_sample_data_{dataset_name}")

    # Split the sample dataset into two and save in tmpdir
    outdir = tmpdir.mkdir(dataset_name)
    split_idx = int(len(data) / 2)
    data[:split_idx].to_parquet(outdir / f"{dataset_name}_1.parquet")
    data[split_idx:].to_parquet(outdir / f"{dataset_name}_2.parquet")
    noising_function = DATASET_GENERATION_FUNCS.get(dataset_name)
    noised_dataset = noising_function(seed=SEED, year=None, source=tmpdir, config=config)

    # Check that shapes and columns are identical
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
def test_seed_behavior(dataset_name: str, config, request):
    """Tests seed behavior"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    noising_function = DATASET_GENERATION_FUNCS.get(dataset_name)
    noised_data_same_seed = noising_function(seed=SEED, year=None, config=config)
    noised_data_different_seed = noising_function(seed=SEED + 1, year=None, config=config)
    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)


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
def test_column_dtypes(dataset_name: str, config, request):
    """Tests that column dtypes are as expected"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, _, _ = _get_common_datasets(dataset_name, data, noised_data)
    config = get_configuration(config)
    for col_name in check_noised.columns:
        col = COLUMNS.get_column(col_name)
        expected_dtype = col.dtype_name
        if expected_dtype == np.dtype(str):
            # str dtype is 'object'
            expected_dtype = np.dtype(object)
        assert noised_data[col.name].dtype == expected_dtype


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
def test_column_noising(dataset_name: str, config, request):
    """Tests that columns are noised as expected"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, check_original, shared_idx = _get_common_datasets(
        dataset_name, data, noised_data
    )
    config = get_configuration(config)
    for col_name in check_noised.columns:
        col = COLUMNS.get_column(col_name)

        # Check that originally missing data remained missing
        originally_missing_idx = check_original.index[check_original[col.name].isna()]
        assert check_noised.loc[originally_missing_idx, col.name].isna().all()

        # Check for noising where applicable
        to_compare_idx = shared_idx.difference(originally_missing_idx)
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
    year = 2030  # not default 2020
    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(year=year)
    assert not data.equals(noised_data)


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        "TODO: tax_1040",
    ],
)
def test_dataset_filter_by_year(mocker, dataset_name: str):
    """Mock the noising function so that it returns the date column of interest
    with the original (unnoised) values to ensure filtering is happening
    """
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(year=year)
    dataset = DATASETS.get_dataset(dataset_name)
    assert (noised_data[dataset.date_column] == year).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.ssa.name,
    ],
)
def test_dataset_filter_by_year_with_full_dates(mocker, dataset_name: str):
    """Mock the noising function so that it returns the date column of interest
    with the original (unnoised) values to ensure filtering is happening
    """
    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(year=year)
    dataset = DATASETS.get_dataset(dataset_name)
    date_format = COLUMNS.get_column(dataset.date_column).additional_attributes.get(
        "date_format"
    )
    if date_format:
        # The date is a string type and we cannot expect to use datetime objects
        # due to the month/day swaps
        years = noised_data[dataset.date_column].str[0:4].astype(int)
    else:
        years = noised_data[dataset.date_column].dt.year
    if dataset == DATASETS.ssa:
        assert (years <= year).all()
    else:
        assert (years == year).all()


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


def _get_common_datasets(dataset_name, data, noised_data):
    """Use unique columns to determine shared non-NA rows between noised and
    unnoised data. Note that we cannot use the original index because that
    gets reset after noising, i.e. the unique columns must NOT be noised.
    """
    # TODO: Replace this with the record ID column when implemented (MIC-4039)
    idx_cols = IDX_COLS.get(dataset_name)
    dataset = DATASETS.get_dataset(dataset_name)
    check_original = _reformat_dates_for_noising(data, dataset).set_index(idx_cols)
    check_noised = noised_data.set_index(idx_cols)
    # Ensure the idx_cols are unique
    assert check_original.index.duplicated().sum() == 0
    assert check_noised.index.duplicated().sum() == 0
    shared_idx = pd.Index(set(check_original.index).intersection(set(check_noised.index)))
    check_original = check_original.loc[shared_idx]
    check_noised = check_noised.loc[shared_idx]
    return check_noised, check_original, shared_idx
