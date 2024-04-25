from pathlib import Path

import pandas as pd
import pytest

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.entities import NO_NOISE
from pseudopeople.constants import paths
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.interface import (
    _coerce_dtypes,
    _reformat_dates_for_noising,
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_1040,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASETS

ROW_PROBABILITY = 0.05
CELL_PROBABILITY = 0.25
SEED = 0
STATE = "RI"

# TODO: Replace this with the record ID column when implemented (MIC-4039)
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
    DATASETS.tax_1040.name: [
        COLUMNS.simulant_id.name,
        COLUMNS.tax_year.name,
    ],
}


@pytest.fixture(scope="session")
def split_sample_data_dir(tmpdir_factory):
    datasets = [
        DatasetNames.CENSUS,
        DatasetNames.ACS,
        DatasetNames.CPS,
        DatasetNames.SSA,
        DatasetNames.TAXES_W2_1099,
        DatasetNames.WIC,
        DatasetNames.TAXES_1040,
    ]
    split_sample_data_dir = tmpdir_factory.mktemp("split_sample_data")
    for dataset_name in datasets:
        data_path = paths.SAMPLE_DATA_ROOT / dataset_name / f"{dataset_name}.parquet"
        data = pd.read_parquet(data_path)
        # Split the sample dataset into two and save in tmpdir_factory
        # We are spliting on household_id as a solution for how to keep households together
        # for the tax 1040 dataset.
        outdir = split_sample_data_dir.mkdir(dataset_name)
        if dataset_name in [
            DatasetNames.TAXES_1040,
            DatasetNames.TAXES_W2_1099,
        ]:
            hh_ids = data.loc[
                data[COLUMNS.tax_year.name] == 2020, COLUMNS.household_id.name
            ].unique()
            hh1_ids = hh_ids[: int(len(hh_ids) / 2)]
            split_household_mask = data[COLUMNS.household_id.name].isin(hh1_ids)
            data[split_household_mask].to_parquet(outdir / f"{dataset_name}_1.parquet")
            data[~split_household_mask].to_parquet(outdir / f"{dataset_name}_2.parquet")
        else:
            split_idx = int(len(data) / 2)
            data[:split_idx].to_parquet(outdir / f"{dataset_name}_1.parquet")
            data[split_idx:].to_parquet(outdir / f"{dataset_name}_2.parquet")

    return Path(split_sample_data_dir)


@pytest.fixture(scope="session")
def split_sample_data_dir_state_edit(tmpdir_factory, split_sample_data_dir):
    # This replace our old tmpdir fixture we were using because this more accurately
    # represents our sample data directory structure with a subdirectory for each
    # dataset and storing all files in the fixture.
    datasets = [
        DatasetNames.CENSUS,
        DatasetNames.ACS,
        DatasetNames.CPS,
        DatasetNames.SSA,
        DatasetNames.TAXES_W2_1099,
        DatasetNames.WIC,
        DatasetNames.TAXES_1040,
    ]
    split_sample_data_dir_state_edit = tmpdir_factory.mktemp("split_sample_data_state_edit")
    for dataset_name in datasets:
        outdir = split_sample_data_dir_state_edit.mkdir(dataset_name)
        data_paths = [
            split_sample_data_dir / dataset_name / f"{dataset_name}_1.parquet",
            split_sample_data_dir / dataset_name / f"{dataset_name}_2.parquet",
        ]
        for data_path in data_paths:
            data = pd.read_parquet(data_path)
            # We do not filter by state for SSA
            if dataset_name != DatasetNames.SSA:
                # Add a state so we can filter for integration tests
                state_column = [column for column in data.columns if "state" in column]
                data.loc[data.reset_index().index % 2 == 0, state_column] = STATE
                data.to_parquet(outdir / data_path.name)

    return Path(split_sample_data_dir_state_edit)


@pytest.fixture(scope="module")
def config():
    """Returns a custom configuration dict to be used in noising"""
    config = get_configuration().to_dict()  # default config

    # Increase row noise probabilities to 5% and column cell_probabilities to 25%
    for dataset_name in config:
        dataset = DATASETS.get_dataset(dataset_name)
        config[dataset.name][Keys.ROW_NOISE] = {
            noise_type.name: {
                Keys.ROW_PROBABILITY: ROW_PROBABILITY,
            }
            for noise_type in dataset.row_noise_types
            if noise_type != NOISE_TYPES.duplicate_with_guardian
        }
        for col in [c for c in dataset.columns if c.noise_types]:
            config[dataset_name][Keys.COLUMN_NOISE][col.name] = {
                noise_type.name: {
                    Keys.CELL_PROBABILITY: CELL_PROBABILITY,
                }
                for noise_type in col.noise_types
            }

    # FIXME: Remove when record_id is added as the truth deck for datasets.
    # For integration tests, we will NOT duplicate rows with guardian duplication.
    # This is because we want to be able to compare the noised and unnoised data
    # and a big assumption we make is that simulant_id and household_id are the
    # truth decks in our datasets.
    config[DATASETS.census.name][Keys.ROW_NOISE][NOISE_TYPES.duplicate_with_guardian.name] = {
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: 0.0,
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: 0.0,
    }
    # Update SSA dataset to noise 'ssn' but NOT noise 'ssa_event_type' since that
    # will be used as an identifier along with simulant_id
    # TODO: Noise ssa_event_type when record IDs are implemented (MIC-4039)
    config[DATASETS.ssa.name][Keys.COLUMN_NOISE][COLUMNS.ssa_event_type.name] = {
        noise_type.name: {
            Keys.CELL_PROBABILITY: 0,
        }
        for noise_type in COLUMNS.ssa_event_type.noise_types
    }
    return config


# Un-noised 1040
@pytest.fixture(scope="session")
def formatted_1040_sample_data():
    formatted_1040 = generate_taxes_1040(seed=SEED, year=None, config=NO_NOISE)
    return formatted_1040


# Noised sample datasets
@pytest.fixture(scope="module")
def noised_sample_data_decennial_census(config):
    return generate_decennial_census(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_american_community_survey(config):
    return generate_american_community_survey(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_current_population_survey(config):
    return generate_current_population_survey(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_women_infants_and_children(config):
    return generate_women_infants_and_children(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_social_security(config):
    return generate_social_security(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_taxes_w2_and_1099(config):
    return generate_taxes_w2_and_1099(seed=SEED, year=None, config=config)


@pytest.fixture(scope="module")
def noised_sample_data_taxes_1040(config):
    return generate_taxes_1040(seed=SEED, year=None, config=config)


# Raw sample datasets with half from a specific state, for state filtering
@pytest.fixture(scope="module")
def sample_data_decennial_census_state_edit():
    data = _load_sample_data(DATASETS.census.name)
    # Set half of the entries to the state we'll filter on
    data.loc[data.reset_index().index % 2 == 0, DATASETS.census.state_column_name] = STATE
    return data


@pytest.fixture(scope="module")
def sample_data_american_community_survey_state_edit():
    data = _load_sample_data(DATASETS.acs.name)
    # Set half of the entries to the state we'll filter on
    data.loc[data.reset_index().index % 2 == 0, DATASETS.acs.state_column_name] = STATE
    return data


@pytest.fixture(scope="module")
def sample_data_current_population_survey_state_edit():
    data = _load_sample_data(DATASETS.cps.name)
    # Set half of the entries to the state we'll filter on
    data.loc[data.reset_index().index % 2 == 0, DATASETS.cps.state_column_name] = STATE
    return data


@pytest.fixture(scope="module")
def sample_data_women_infants_and_children_state_edit():
    data = _load_sample_data(DATASETS.wic.name)
    # Set half of the entries to the state we'll filter on
    data.loc[data.reset_index().index % 2 == 0, DATASETS.wic.state_column_name] = STATE
    return data


@pytest.fixture(scope="module")
def sample_data_taxes_w2_and_1099_state_edit():
    data = _load_sample_data(DATASETS.tax_w2_1099.name)
    # Set half of the entries to the state we'll filter on
    data.loc[
        data.reset_index().index % 2 == 0, DATASETS.tax_w2_1099.state_column_name
    ] = STATE
    return data


####################
# HELPER FUNCTIONS #
####################


def _load_sample_data(dataset, request):
    if dataset == DatasetNames.TAXES_1040:
        # We need to get formatted 1040 data that is not noised to get the expected columns
        data = request.getfixturevalue("formatted_1040_sample_data")
    else:
        data_path = paths.SAMPLE_DATA_ROOT / dataset / f"{dataset}.parquet"
        data = pd.read_parquet(data_path)

    return data


def _get_common_datasets(dataset_name, data, noised_data):
    """Use unique columns to determine shared non-NA rows between noised and
    unnoised data. Note that we cannot use the original index because that
    gets reset after noising, i.e. the unique columns must NOT be noised.
    """
    idx_cols = IDX_COLS.get(dataset_name)
    dataset = DATASETS.get_dataset(dataset_name)
    check_original = _coerce_dtypes(
        _reformat_dates_for_noising(data, dataset), dataset
    ).set_index(idx_cols)
    check_noised = noised_data.set_index(idx_cols)
    # Ensure the idx_cols are unique
    assert check_original.index.duplicated().sum() == 0
    assert check_noised.index.duplicated().sum() == 0
    shared_idx = pd.Index(set(check_original.index).intersection(set(check_noised.index)))
    check_original = check_original.loc[shared_idx]
    check_noised = check_noised.loc[shared_idx]
    return check_noised, check_original, shared_idx
