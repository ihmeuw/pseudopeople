import pytest
from layered_config_tree import LayeredConfigTree

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.entity_types import RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASET_SCHEMAS
from tests.integration.conftest import _initialize_dataset_with_sample


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
        DATASET_SCHEMAS.ssa.name,
        DATASET_SCHEMAS.tax_w2_1099.name,
        DATASET_SCHEMAS.wic.name,
        DATASET_SCHEMAS.tax_1040.name,
    ],
)
def test_dataset_missingness(dataset_name: str):
    """Tests that missingness is accurate with dataset.data."""
    dataset = _initialize_dataset_with_sample(dataset_name)
    # We must manually clean the data for noising since we are recreating our main noising loop
    dataset._clean_input_data()
    dataset._reformat_dates_for_noising()
    config = get_configuration()
    dataset_config = config[dataset_name]
    # NOTE: This is recreating Dataset._noise_dataset but adding assertions for missingness
    for noise_type in NOISE_TYPES:
        if isinstance(noise_type, RowNoiseType):
            if noise_type.name not in dataset_config[Keys.ROW_NOISE]:
                continue
            else:
                row_noise_config: LayeredConfigTree = dataset_config[Keys.ROW_NOISE][
                    noise_type.name
                ]
                noise_type(dataset, row_noise_config)
                # Check missingness is synced with data
                assert dataset.missingness.equals(dataset.is_missing(dataset.data))
        else:
            column_noise_config: LayeredConfigTree = dataset_config[Keys.COLUMN_NOISE]
            columns_to_noise = [
                col
                for col in column_noise_config
                if col in dataset.data.columns and noise_type.name in column_noise_config[col]
            ]
            for column in columns_to_noise:
                column_noise_config: LayeredConfigTree = dataset_config[Keys.COLUMN_NOISE][
                    column
                ][noise_type.name]
                noise_type(
                    dataset,
                    column_noise_config[column][noise_type.name],
                    column,
                )
                # Check missingness is synced with data
                assert dataset.missingness.equals(dataset.is_missing(dataset.data))
