from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd
import pytest
from _pytest.fixtures import FixtureRequest
from vivarium_testing_utils import FuzzyChecker

from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_1040,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASET_SCHEMAS
from pseudopeople.utilities import (
    count_number_of_tokens_per_string,
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
)
from tests.integration.conftest import (
    IDX_COLS,
    _get_common_datasets,
    get_unnoised_data,
)
from tests.utilities import (
    initialize_dataset_with_sample,
    run_column_noising_tests,
    run_omit_row_or_do_not_respond_tests,
)

DATASET_GENERATION_FUNCS: dict[str, Callable[..., Any]] = {
    DATASET_SCHEMAS.census.name: generate_decennial_census,
    DATASET_SCHEMAS.acs.name: generate_american_community_survey,
    DATASET_SCHEMAS.cps.name: generate_current_population_survey,
    DATASET_SCHEMAS.ssa.name: generate_social_security,
    DATASET_SCHEMAS.tax_w2_1099.name: generate_taxes_w2_and_1099,
    DATASET_SCHEMAS.wic.name: generate_women_infants_and_children,
    DATASET_SCHEMAS.tax_1040.name: generate_taxes_1040,
}

TOKENS_PER_STRING_MAPPER: dict[str, Callable[..., pd.Series[int]]] = {
    NOISE_TYPES.make_ocr_errors.name: partial(
        count_number_of_tokens_per_string, pd.Series(load_ocr_errors().index)
    ),
    NOISE_TYPES.make_phonetic_errors.name: partial(
        count_number_of_tokens_per_string,
        pd.Series(load_phonetic_errors().index),
    ),
    NOISE_TYPES.write_wrong_digits.name: lambda x: x.astype(str)
    .str.replace(r"[^\d]", "", regex=True)
    .str.len(),
    NOISE_TYPES.make_typos.name: partial(
        count_number_of_tokens_per_string, pd.Series(load_qwerty_errors_data().index)
    ),
}


def test_column_noising(
    config: dict[str, Any],
    dataset_name: str,
    request: FixtureRequest,
    fuzzy_checker: FuzzyChecker,
) -> None:
    """Tests that columns are noised as expected"""
    original = request.getfixturevalue("unnoised_dataset")
    noised_data = request.getfixturevalue("data")

    check_noised, check_original, shared_idx = _get_common_datasets(original, noised_data)

    run_column_noising_tests(
        dataset_name, config, fuzzy_checker, check_noised, check_original, shared_idx
    )


def test_row_noising_omit_row_or_do_not_respond(
    dataset_name: str, config: dict[str, Any], request: FixtureRequest
) -> None:
    """Tests that omit_row and do_not_respond row noising are being applied"""
    idx_cols = IDX_COLS.get(dataset_name)
    original = get_unnoised_data(dataset_name)
    original_data = original.data.set_index(idx_cols)
    noised_data = request.getfixturevalue("data")
    noised_data = noised_data.set_index(idx_cols)

    run_omit_row_or_do_not_respond_tests(dataset_name, config, original_data, noised_data)


def test_unnoised_id_cols(dataset_name: str, request: FixtureRequest) -> None:
    """Tests that all datasets retain unnoised simulant_id and household_id
    (except for SSA which does not include household_id)
    """
    unnoised_id_cols = [COLUMNS.simulant_id.name]
    if dataset_name != DATASET_SCHEMAS.ssa.name:
        unnoised_id_cols.append(COLUMNS.household_id.name)
    original = initialize_dataset_with_sample(dataset_name)
    noised_data = request.getfixturevalue("data")
    check_noised, check_original, _ = _get_common_datasets(original, noised_data)
    assert (
        (
            check_original.reset_index()[unnoised_id_cols]
            == check_noised.reset_index()[unnoised_id_cols]
        )
        .all()
        .all()
    )
