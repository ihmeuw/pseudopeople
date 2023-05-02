from typing import Callable

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants import metadata, paths
from pseudopeople.interface import (
    _reformat_dates_for_noising,
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.schema_entities import COLUMNS, DATASETS, NOISE_TYPES, Dataset

# TODO: Move into a metadata file and import metadata into prl
DATA_COLUMNS = ["year", "event_date", "survey_date", "tax_year"]

# TODO [MIC-4038]: Refactor all of these noised datasets into module-level fixtures


# TODO [MIC-4040]: Break this up into multiple different tests once MIC-4038 is
# complete (creates fixtures out of the noised data so we don't have to
# re-noise the same stuff over and over)
@pytest.mark.parametrize(
    "dataset, noising_function, use_sample_data",
    [
        (DATASETS.census, generate_decennial_census, True),
        (DATASETS.census, generate_decennial_census, False),
        (DATASETS.acs, generate_american_community_survey, True),
        (DATASETS.acs, generate_american_community_survey, False),
        (DATASETS.cps, generate_current_population_survey, True),
        (DATASETS.cps, generate_current_population_survey, False),
        (DATASETS.ssa, generate_social_security, True),
        (DATASETS.ssa, generate_social_security, False),
        (DATASETS.tax_w2_1099, generate_taxes_w2_and_1099, True),
        (DATASETS.tax_w2_1099, generate_taxes_w2_and_1099, False),
        (DATASETS.wic, generate_women_infants_and_children, True),
        (DATASETS.wic, generate_women_infants_and_children, False),
        ("DATASETS.tax_1040", "todo", True),
        ("DATASETS.tax_1040", "todo", False),
    ],
)
def test_generate_dataset_and_col_noising(
    dataset: Dataset, noising_function: Callable, use_sample_data: bool, tmpdir, mocker
):
    """Tests that noised datasets are generated and as expected. The 'use_sample_data'
    parameter determines whether to use the sample data (if True) or
    a non-default root directory with multiple datasets to compile (if False)
    """
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement dataset {dataset}")

    sample_data_path = list((paths.SAMPLE_DATA_ROOT / dataset.name).glob(f"{dataset.name}*"))[
        0
    ]

    data = _load_data(sample_data_path)

    # Configure if default (sample data) is used or a different root directory
    if use_sample_data:
        source = None  # will default to using sample data
    else:
        source = _generate_non_default_data_root(dataset.name, tmpdir, sample_data_path, data)

    # Increase cell probability
    cell_probability = 0.25
    custom_config = {dataset.name: {Keys.COLUMN_NOISE: {}}}  # initialize
    for col in [c for c in dataset.columns if c.noise_types]:
        custom_config[dataset.name][Keys.COLUMN_NOISE][col.name] = {
            noise_type.name: {
                Keys.CELL_PROBABILITY: cell_probability,
            }
            for noise_type in col.noise_types
        }

    # Update SSA dataset to noise 'ssn' but NOT noise 'ssa_event_type' since that
    # will be used as an identifier along with simulant_id
    # TODO: Noise ssa_event_type when record IDs are implemented (MIC-4039)
    if dataset.name == DATASETS.ssa.name:
        custom_config[dataset.name][Keys.COLUMN_NOISE][COLUMNS.ssa_event_type.name] = {
            noise_type.name: {
                Keys.CELL_PROBABILITY: 0,
            }
            for noise_type in COLUMNS.ssa_event_type.noise_types
        }

    noised_data = noising_function(seed=0, source=source, year=None, config=custom_config)
    noised_data_same_seed = noising_function(
        seed=0, source=source, year=None, config=custom_config
    )
    noised_data_different_seed = noising_function(
        seed=1, source=source, year=None, config=custom_config
    )

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)

    # Check each column. We set the index for each dataset to be unique
    # identifiers b/c the original index gets reset after noising. Note that
    # the uniquely identifying columns must NOT be noised.
    # TODO: Replace this with the record ID column when implemented (MIC-4039)
    idx_cols = {
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
    }.get(dataset.name)
    check_original = _reformat_dates_for_noising(data, dataset).set_index(idx_cols)
    check_noised = noised_data.set_index(idx_cols)
    assert check_original.index.duplicated().sum() == 0
    assert check_noised.index.duplicated().sum() == 0
    shared_idx = pd.Index(set(check_original.index).intersection(set(check_noised.index)))
    check_original = check_original.loc[shared_idx]
    check_noised = check_noised.loc[shared_idx]

    config = get_configuration(custom_config)
    for col_name in check_noised.columns:
        col = COLUMNS.get_column(col_name)
        # Check dtype is correct
        expected_dtype = col.dtype_name
        if expected_dtype == np.dtype(str):
            # str dtype is 'object'
            expected_dtype = np.dtype(object)
        assert noised_data[col_name].dtype == expected_dtype

        # Check that originally missing data remained missing
        originally_missing_idx = check_original.index[check_original[col_name].isna()]
        assert check_noised.loc[originally_missing_idx, col_name].isna().all()

        # Check for noising where applicable
        to_compare_idx = shared_idx.difference(originally_missing_idx)
        if col.noise_types:
            assert (
                check_original.loc[to_compare_idx, col_name].values
                != check_noised.loc[to_compare_idx, col_name].values
            ).any()

            noise_level = (
                check_original.loc[to_compare_idx, col_name].values
                != check_noised.loc[to_compare_idx, col_name].values
            ).mean()

            # Check that the amount of noising seems reasonable
            tmp_config = config[dataset.name][Keys.COLUMN_NOISE][col_name]
            includes_token_noising = [
                c for c in tmp_config if Keys.TOKEN_PROBABILITY in tmp_config[c].keys()
            ]
            # NOTE: The threshold assigned when we have token-level noising
            # is guessed at since it's difficult to calculate.
            # TODO: Come up with a more accurate values. There are token probabilities
            # and additional parameters to consider as well as the rtol when the
            # number of compared is small.
            expected_noise = 1 - (1 - cell_probability) ** len(col.noise_types)
            rtol = 0.5 if includes_token_noising else 0.11
            assert np.isclose(noise_level, expected_noise, rtol=rtol)
        else:  # No noising - should be identical
            assert (
                check_original.loc[to_compare_idx, col_name].values
                == check_noised.loc[to_compare_idx, col_name].values
            ).all()


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


# TODO: Test against multiple datasets being loaded and concated
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
    year = 2030  # not default 2020
    data_path = paths.SAMPLE_DATA_ROOT / data_dir_name / f"{data_dir_name}.parquet"
    data = _load_data(data_path)

    noised_data = noising_function(year=year, seed=0)
    noised_data_same_seed = noising_function(year=year, seed=0)
    noised_data_different_seed = noising_function(year=year, seed=1)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)


def _mock_extract_columns(columns_to_keep, noised_dataset):
    return noised_dataset


# TODO: Test against multiple datasets being loaded and concated
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

    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = noising_function(year=year)

    assert (noised_data[date_column] == year).all()


def _mock_noise_dataset(
    dataset,
    dataset_data: pd.DataFrame,
    configuration,
    seed: int,
):
    """Mock noise_dataset that just returns unnoised data"""
    return dataset_data


# TODO: Test against multiple datasets being loaded and concated
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
    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = noising_function(year=year)

    dates = pd.DatetimeIndex(noised_data[dataset.date_column])
    if dataset == DATASETS.ssa:
        assert (dates.year <= year).all()
    else:
        assert (dates.year == year).all()


def _load_data(data_path):
    if data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)
    elif data_path.suffix == ".hdf":
        data = pd.read_hdf(data_path)
    else:
        raise NotImplementedError(f"Expected hdf or parquet but got {data_path.suffix}")

    return data
