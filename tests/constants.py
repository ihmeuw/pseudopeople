from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd

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
from pseudopeople.schema_entities import DATASET_SCHEMAS
from pseudopeople.utilities import (
    count_number_of_tokens_per_string,
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
)

CELL_PROBABILITY = 0.25

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
