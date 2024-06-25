import numpy as np
import pandas as pd
import pytest

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.entity_types import ColumnNoiseType, NoiseType, RowNoiseType
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
    # Tests that missingness is accurate with dataset.data
    # mocker.patch("pseudopeople.interface.validate_source_compatibility")
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
                noise_type(dataset, dataset_config[Keys.ROW_NOISE][noise_type.name])
                # Check missingness is synced with data
                assert dataset.missingness.equals(dataset.is_missing(dataset.data))
        else:
            columns_to_noise = [
                col
                for col in dataset_config[Keys.COLUMN_NOISE]
                if col in dataset.data.columns
                and noise_type.name in dataset_config[Keys.COLUMN_NOISE][col]
            ]
            for column in columns_to_noise:
                noise_type(
                    dataset,
                    dataset_config[Keys.COLUMN_NOISE][column][noise_type.name],
                    column,
                )
                # Check missingness is synced with data
                assert dataset.missingness.equals(dataset.is_missing(dataset.data))
