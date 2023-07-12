from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASETS
from tests.integration.conftest import (
    CELL_PROBABILITY,
    IDX_COLS,
    SEED,
    STATE,
    _get_common_datasets,
)

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
def test_generate_dataset_from_sample_and_source(
    dataset_name: str, user_config, tmpdir, request
):
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
    # Generate a new (non-fixture) noised dataset from the split data in tmpdir
    noising_function = DATASET_GENERATION_FUNCS.get(dataset_name)
    noised_dataset = noising_function(
        seed=SEED, year=None, source=Path(tmpdir), config=user_config
    )

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
        noise_level_single_dataset = (
            check_original.loc[compare_sample_idx, col].values
            != check_noised_sample.loc[compare_sample_idx, col].values
        ).mean()
        noise_level_full_dataset = (
            check_original.loc[compare_dataset_idx, col].values
            != check_noised_dataset.loc[compare_dataset_idx, col].values
        ).mean()
        # we special-case a few sparse columns that have larger differences
        if dataset_name == DATASETS.cps.name and col == COLUMNS.unit_number.name:
            rtol = 0.30
        elif dataset_name == DATASETS.acs.name and col == COLUMNS.unit_number.name:
            rtol = 0.25
        else:
            rtol = 0.12
        assert np.isclose(noise_level_full_dataset, noise_level_single_dataset, rtol=rtol)


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
def test_seed_behavior(dataset_name: str, user_config, request):
    """Tests seed behavior"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    # Generate new (non-fixture) noised datasets with the same seed and a different
    # seed as the fixture
    noising_function = DATASET_GENERATION_FUNCS.get(dataset_name)
    noised_data_same_seed = noising_function(seed=SEED, year=None, config=user_config)
    noised_data_different_seed = noising_function(
        seed=SEED + 1, year=None, config=user_config
    )
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
def test_column_dtypes(dataset_name: str, request):
    """Tests that column dtypes are as expected"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, _, _ = _get_common_datasets(dataset_name, data, noised_data)
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
def test_column_noising(dataset_name: str, user_config, request):
    """Tests that columns are noised as expected"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}")
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, check_original, shared_idx = _get_common_datasets(
        dataset_name, data, noised_data
    )
    config = get_configuration(user_config)
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
            rtol = 0.70 if includes_token_noising else 0.11
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
def test_row_noising_omit_row_or_do_not_respond(dataset_name: str, user_config, request):
    """Tests that omit_row and do_not_respond row noising are being applied"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    idx_cols = IDX_COLS.get(dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}").set_index(idx_cols)
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}").set_index(
        idx_cols
    )
    config = get_configuration(user_config)[dataset_name][Keys.ROW_NOISE]
    noise_type = [
        n for n in config if n in [NOISE_TYPES.omit_row.name, NOISE_TYPES.do_not_respond.name]
    ]
    assert len(noise_type) < 2  # omit_row and do_not_respond should be mutually exclusive
    if not noise_type:  # Check that there are no missing indexes
        assert noised_data.index.symmetric_difference(data.index).empty
    else:  # Check that there are some omissions
        # TODO: assert levels are as expected
        assert noised_data.index.difference(data.index).empty
        assert not data.index.difference(noised_data.index).empty


@pytest.mark.skip(reason="TODO: Implement duplication row noising")
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
def test_row_noising_duplication(dataset_name: str, user_config, request):
    """Tests that duplication row noising is being applied"""
    ...


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
    # Generate a new (non-fixture) noised dataset for a single year
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
    # Generate a new (non-fixture) noised dataset for a single year but mocked such
    # that no noise actually happens (otherwise the years would get noised and
    # we couldn't tell if the filter was working properly)
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(year=year)
    dataset = DATASETS.get_dataset(dataset_name)
    assert (noised_data[dataset.date_column_name] == year).all()


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
    # Generate a new (non-fixture) noised dataset for a single year but mocked such
    # that no noise actually happens (otherwise the years would get noised and
    # we couldn't tell if the filter was working properly)
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(year=year)
    dataset = DATASETS.get_dataset(dataset_name)
    # NB: conditional on date_format is potentially brittle if we have some noised
    #  date column with a different format.
    date_format = COLUMNS.get_column(dataset.date_column_name).additional_attributes.get(
        "date_format"
    )
    if date_format:
        # The date is a string type, and we cannot expect to use datetime objects
        # due to the month/day swaps
        years = noised_data[dataset.date_column_name].str[0:4].astype(int)
    else:
        years = noised_data[dataset.date_column_name].dt.year
    if dataset == DATASETS.ssa:
        assert (years <= year).all()
    else:
        assert (years == year).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        "TODO: tax_1040",
    ],
)
def test_generate_dataset_with_state_noised(dataset_name: str, request, tmpdir):
    """Test that dataset generation filtered by state noises data differently with different seeds"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}_state_edit")
    _generate_non_sample_data_root(dataset_name, tmpdir, data)

    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(source=Path(tmpdir), seed=0, state=STATE)
    noised_data_same_seed = noising_function(source=Path(tmpdir), seed=0, state=STATE)
    noised_data_different_seed = noising_function(source=Path(tmpdir), seed=1, state=STATE)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        "TODO: tax_1040",
    ],
)
def test_generate_dataset_with_state_filtered(dataset_name: str, request, mocker, tmpdir):
    """Test that values returned by dataset generators are only for the specified state"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    dataset = DATASETS.get_dataset(dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}_state_edit")
    _generate_non_sample_data_root(dataset_name, tmpdir, data)

    # Skip noising (noising can incorrect select another state)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)

    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(source=Path(tmpdir), state=STATE)

    assert (noised_data[dataset.state_column_name] == STATE).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        "TODO: tax_1040",
    ],
)
def test_generate_dataset_with_state_unfiltered(dataset_name: str, request, mocker, tmpdir):
    """Test that values returned by dataset generators are for all locations if state unspecified"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    dataset = DATASETS.get_dataset(dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}_state_edit")
    _generate_non_sample_data_root(dataset_name, tmpdir, data)

    # Skip noising (noising can incorrect select another state)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)

    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(source=Path(tmpdir))

    assert len(noised_data[dataset.state_column_name].unique()) > 1


@pytest.mark.parametrize(
    "dataset_name",
    [DATASETS.census.name, DATASETS.tax_w2_1099.name, DATASETS.wic.name, "TODO: tax_1040"],
)
def test_dataset_filter_by_state_and_year(mocker, tmpdir, request, dataset_name: str):
    """Test that dataset generation works with state and year filters in conjunction"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    year = 2030  # not default 2020
    data = request.getfixturevalue(f"sample_data_{dataset_name}_state_edit")
    _generate_non_sample_data_root(dataset_name, tmpdir, data)
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(source=Path(tmpdir), year=year, state=STATE)
    dataset = DATASETS.get_dataset(dataset_name)
    assert (noised_data[dataset.date_column_name] == year).all()
    assert (noised_data[dataset.state_column_name] == STATE).all()


@pytest.mark.parametrize(
    "dataset_name",
    [DATASETS.acs.name, DATASETS.cps.name],
)
def test_dataset_filter_by_state_and_year_with_full_dates(
    mocker, tmpdir, request, dataset_name: str
):
    """Test that dataset generation works with state and year filters in conjunction"""
    year = 2030  # not default 2020
    data = request.getfixturevalue(f"sample_data_{dataset_name}_state_edit")
    _generate_non_sample_data_root(dataset_name, tmpdir, data)
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    noised_data = noising_function(source=Path(tmpdir), year=year, state=STATE)
    dataset = DATASETS.get_dataset(dataset_name)
    # NB: conditional on date_format is potentially brittle if we have some noised
    #  date column with a different format.
    date_format = COLUMNS.get_column(dataset.date_column_name).additional_attributes.get(
        "date_format"
    )
    if date_format:
        # The date is a string type, and we cannot expect to use datetime objects
        # due to the month/day swaps
        years = noised_data[dataset.date_column_name].str[0:4].astype(int)
    else:
        years = noised_data[dataset.date_column_name].dt.year
    assert (years == year).all()
    assert (noised_data[dataset.state_column_name] == STATE).all()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        "TODO: tax_1040",
    ],
)
def test_generate_dataset_with_bad_state(dataset_name: str, request, tmpdir):
    """Test that bad state values result in informative ValueErrors"""
    if "TODO" in dataset_name:
        pytest.skip(reason=dataset_name)
    data = request.getfixturevalue(f"sample_data_{dataset_name}_state_edit")
    _generate_non_sample_data_root(dataset_name, tmpdir, data)
    bad_state = "Silly State That Doesn't Exist"

    noising_function = DATASET_GENERATION_FUNCS[dataset_name]
    with pytest.raises(ValueError, match=bad_state.upper()):
        _ = noising_function(source=Path(tmpdir), state=bad_state)


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


def _generate_non_sample_data_root(data_dir_name, tmpdir, data):
    """Helper function to break the single sample dataset into two and save
    out to tmpdir to be used as a non-default 'source' argument
    """
    outdir = tmpdir.mkdir(data_dir_name)
    split_idx = int(len(data) / 2)
    data[:split_idx].to_parquet(outdir / f"{data_dir_name}_1.parquet")
    data[split_idx:].to_parquet(outdir / f"{data_dir_name}_2.parquet")
    return tmpdir
