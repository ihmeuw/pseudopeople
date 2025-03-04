from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from layered_config_tree import LayeredConfigTree
from pytest_check import check
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.configuration.noise_configuration import NoiseConfiguration
from pseudopeople.dataset import Dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from tests.integration.conftest import IDX_COLS, _get_common_datasets, get_unnoised_data
from tests.utilities import initialize_dataset_with_sample, run_column_noising_tests


def test_column_noising(
    unnoised_dataset: Dataset,
    noised_data: pd.DataFrame,
    config: dict[str, Any],
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that columns are noised as expected"""
    check_noised, check_original, shared_idx = _get_common_datasets(
        unnoised_dataset, noised_data
    )

    run_column_noising_tests(
        dataset_name, config, fuzzy_checker, check_noised, check_original, shared_idx
    )


def test_row_noising_omit_row_or_do_not_respond(
    dataset_name: str,
    config: dict[str, Any],
) -> None:
    """Tests that the correct noising is applied to each dataset when
    noising with omit_row and do_not_respond by checking the config."""
    noise_config: NoiseConfiguration = get_configuration(config)
    noise_types = [
        noise_type
        for noise_type in [NOISE_TYPES.omit_row.name, NOISE_TYPES.do_not_respond.name]
        if noise_config.has_noise_type(dataset_name, noise_type)
    ]

    if dataset_name in [
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.cps.name,
    ]:
        # Census and household surveys have do_not_respond and omit_row.
        # For all other datasets they are mutually exclusive
        with check:
            assert len(noise_types) == 2
    else:
        with check:
            assert len(noise_types) < 2


def get_single_noise_type_config(
    dataset_name: str, noise_type_to_keep: str
) -> dict[str, Any]:
    """Return a NoiseConfiguration object with no noising except for noise_type_to_keep,
    which will contain the default values from get_configuration."""
    config: NoiseConfiguration = get_configuration()
    config_dict = config.to_dict()

    for noise_type, probabilities in config_dict[dataset_name][Keys.ROW_NOISE].items():
        if noise_type != noise_type_to_keep:
            for probability_name, probability in probabilities.items():
                config_dict[dataset_name][Keys.ROW_NOISE][noise_type][probability_name] = 0.0

    for col, noise_types in config_dict[dataset_name][Keys.COLUMN_NOISE].items():
        for noise_type, probabilities in noise_types.items():
            if noise_type != noise_type_to_keep:
                for probability_name, probability in probabilities.items():
                    if isinstance(probability, list):
                        new_probability = [0.0 for x in probability]
                    elif isinstance(probability, dict):
                        new_probability = {key: 0.0 for key in probability.keys()}
                    else:
                        new_probability = 0.0
                    config_dict[dataset_name][Keys.COLUMN_NOISE][col][noise_type][probability_name] = new_probability

    return config_dict


@pytest.mark.parametrize("expected_noise", ["default", 0.01])
def test_omit_row(
    expected_noise: str | float,
    unnoised_dataset: Dataset,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that omit_row noising is being applied at the expected proportion."""
    original_data = unnoised_dataset.data
    config_dict = get_single_noise_type_config(dataset_name, NOISE_TYPES.omit_row.name)

    if expected_noise != "default":
        config_dict[dataset_name][Keys.ROW_NOISE][NOISE_TYPES.omit_row.name][Keys.ROW_PROBABILITY] = expected_noise
        config = NoiseConfiguration(LayeredConfigTree(config_dict))
    else:
        config = NoiseConfiguration(LayeredConfigTree(config_dict))
        expected_noise: float = config.get_row_probability(
            dataset_name, NOISE_TYPES.omit_row.name
        )
    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    dataset = Dataset(dataset_schema, original_data, 0)
    NOISE_TYPES.omit_row(dataset, config)
    noised_data = dataset.data

    fuzzy_checker.fuzzy_assert_proportion(
        name="test_omit_row",
        observed_numerator=len(noised_data),
        observed_denominator=len(original_data),
        target_proportion=1 - expected_noise,
    )

    assert set(noised_data.columns) == set(original_data.columns)
    assert (noised_data.dtypes == original_data.dtypes).all()


@pytest.mark.parametrize("expected_noise", ["default", 0.03])
def test_do_not_respond(
    expected_noise: str | float,
    unnoised_dataset: Dataset,
    dataset_name: str,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that do_not_respond noising is being applied at the expected proportion."""
    if dataset_name not in [
        DATASET_SCHEMAS.acs.name,
        DATASET_SCHEMAS.census.name,
        DATASET_SCHEMAS.cps.name,
    ]:
        return

    original_data = unnoised_dataset.data
    config_dict = get_single_noise_type_config(dataset_name, NOISE_TYPES.do_not_respond.name)

    if expected_noise != "default":
        config_dict[dataset_name][Keys.ROW_NOISE][NOISE_TYPES.do_not_respond.name][Keys.ROW_PROBABILITY] = expected_noise
        config = NoiseConfiguration(LayeredConfigTree(config_dict))
    else:
        config = NoiseConfiguration(LayeredConfigTree(config_dict))
        expected_noise: float = config.get_row_probability(
            dataset_name, NOISE_TYPES.do_not_respond.name
        )

    dataset_schema = DATASET_SCHEMAS.get_dataset_schema(dataset_name)
    dataset = Dataset(dataset_schema, original_data, 0)
    NOISE_TYPES.do_not_respond(dataset, config)
    noised_data = dataset.data

    # Check ACS and CPS data is scaled properly due to oversampling
    if dataset_name is not DATASET_SCHEMAS.census.name:  # is acs or cps
        expected_noise = 0.5 + expected_noise / 2

    # Test that noising affects expected proportion with expected types
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_do_not_respond",
        observed_numerator=len(original_data) - len(noised_data),
        observed_denominator=len(original_data),
        target_proportion=expected_noise,
        name_additional=f"noised_data",
    )
    assert set(noised_data.columns) == set(original_data.columns)
    assert (noised_data.dtypes == original_data.dtypes).all()


def test_column_dtypes(
    unnoised_dataset: Dataset,
    noised_data: pd.DataFrame,
    dataset_name: str,
    config: dict[str, Any],
) -> None:
    """Tests that column dtypes are as expected"""
    for col_name in noised_data.columns:
        col = COLUMNS.get_column(col_name)
        expected_dtype = col.dtype_name
        if expected_dtype == np.dtype(object):
            # str dtype is 'object'
            # Check that they are actually strings and not some other
            # type of object.
            actual_types = noised_data[col.name].dropna().apply(type)
            assert (actual_types == str).all(), actual_types.unique()
        assert noised_data[col.name].dtype == expected_dtype


def test_unnoised_id_cols(dataset_name: str, request: FixtureRequest) -> None:
    """Tests that all datasets retain unnoised simulant_id and household_id
    (except for SSA which does not include household_id)
    """
    unnoised_id_cols = [COLUMNS.simulant_id.name]
    if dataset_name != DATASET_SCHEMAS.ssa.name:
        unnoised_id_cols.append(COLUMNS.household_id.name)
    original = initialize_dataset_with_sample(dataset_name)
    noised_data = request.getfixturevalue("noised_data")
    check_noised, check_original, _ = _get_common_datasets(original, noised_data)
    assert (
        (
            check_original.reset_index()[unnoised_id_cols]
            == check_noised.reset_index()[unnoised_id_cols]
        )
        .all()
        .all()
    )
