import pandas as pd
import pytest

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants import paths
from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.schema_entities import COLUMNS, DATASETS

CELL_PROBABILITY = 0.25
SEED = 0
STATE = "RI"


@pytest.fixture(scope="module")
def config():
    """Returns a custom configuration to be used in noising"""
    config = get_configuration().to_dict()  # default config

    # Increase cell_probability to 25% to ensure we noise spare columns
    for dataset_name in config:
        dataset = DATASETS.get_dataset(dataset_name)
        for col in [c for c in dataset.columns if c.noise_types]:
            config[dataset_name][Keys.COLUMN_NOISE][col.name] = {
                noise_type.name: {
                    Keys.CELL_PROBABILITY: CELL_PROBABILITY,
                }
                for noise_type in col.noise_types
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


# Raw sample datasets
@pytest.fixture(scope="module")
def sample_data_decennial_census():
    return _load_sample_data("decennial_census")


@pytest.fixture(scope="module")
def sample_data_american_community_survey():
    return _load_sample_data("american_community_survey")


@pytest.fixture(scope="module")
def sample_data_current_population_survey():
    return _load_sample_data("current_population_survey")


@pytest.fixture(scope="module")
def sample_data_women_infants_and_children():
    return _load_sample_data("women_infants_and_children")


@pytest.fixture(scope="module")
def sample_data_social_security():
    return _load_sample_data("social_security")


@pytest.fixture(scope="module")
def sample_data_taxes_w2_and_1099():
    return _load_sample_data("taxes_w2_and_1099")


@pytest.fixture(scope="module")
def sample_data_taxes_1040():
    return _load_sample_data("taxes_1040")


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


# @pytest.fixture(scope="module")
# def noised_sample_data_taxes_1040(config):
#     return generate_taxes_1040(seed=SEED, year=None, config=config)


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


def _load_sample_data(dataset):
    data_path = paths.SAMPLE_DATA_ROOT / dataset / f"{dataset}.parquet"
    return pd.read_parquet(data_path)
