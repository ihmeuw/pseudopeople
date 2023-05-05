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

ROW_PROBABILITY = 0.05
CELL_PROBABILITY = 0.25
SEED = 0


@pytest.fixture(scope="module")
def user_config():
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
        }
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
def noised_sample_data_decennial_census(user_config):
    return generate_decennial_census(seed=SEED, year=None, config=user_config)


@pytest.fixture(scope="module")
def noised_sample_data_american_community_survey(user_config):
    return generate_american_community_survey(seed=SEED, year=None, config=user_config)


@pytest.fixture(scope="module")
def noised_sample_data_current_population_survey(user_config):
    return generate_current_population_survey(seed=SEED, year=None, config=user_config)


@pytest.fixture(scope="module")
def noised_sample_data_women_infants_and_children(user_config):
    return generate_women_infants_and_children(seed=SEED, year=None, config=user_config)


@pytest.fixture(scope="module")
def noised_sample_data_social_security(user_config):
    return generate_social_security(seed=SEED, year=None, config=user_config)


@pytest.fixture(scope="module")
def noised_sample_data_taxes_w2_and_1099(user_config):
    return generate_taxes_w2_and_1099(seed=SEED, year=None, config=user_config)


# @pytest.fixture(scope="module")
# def noised_sample_data_taxes_1040(config):
#     return generate_taxes_1040(seed=SEED, year=None, config=user_config)


####################
# HELPER FUNCTIONS #
####################


def _load_sample_data(dataset):
    data_path = paths.SAMPLE_DATA_ROOT / dataset / f"{dataset}.parquet"
    return pd.read_parquet(data_path)
