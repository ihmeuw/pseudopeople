from typing import NamedTuple

from pseudopeople import noise_functions
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType


class __NoiseTypes(NamedTuple):
    """Container for all noise types in the order in which they should be applied:
    omissions, duplications, missing data, incorrect selection, copy from w/in
    household, month and day swaps, zip code miswriting, age miswriting,
    numeric miswriting, nicknames, fake names, phonetic, OCR, typographic"

    NOTE: Any configuration tree overwrites in these objects are what ends up
    in the "baseline" ConfigTree layer.
    """

    OMISSION: RowNoiseType = RowNoiseType("omission", noise_functions.omit_rows, is_implemented=False)  # TODO
    DUPLICATION: RowNoiseType = RowNoiseType("duplication", noise_functions.duplicate_rows, is_implemented=False)  # TODO
    MISSING_DATA: ColumnNoiseType = ColumnNoiseType(
        "missing_data",
        noise_functions.generate_missing_data,
        token_noise_level=None,
    )
    INCORRECT_SELECTION: ColumnNoiseType = ColumnNoiseType(
        "incorrect_selection",
        noise_functions.generate_incorrect_selections,
        token_noise_level=None,
    )
    COPY_FROM_WITHIN_HOUSEHOLD: ColumnNoiseType = ColumnNoiseType(
        "copy_from_within_household", noise_functions.generate_within_household_copies, is_implemented=False,
    )  # TODO
    MONTH_DAY_SWAP: ColumnNoiseType = ColumnNoiseType(
        "month_day_swap", noise_functions.swap_months_and_days, is_implemented=False,
    )  # TODO
    ZIPCODE_MISWRITING: ColumnNoiseType = ColumnNoiseType(
        "zipcode_miswriting", noise_functions.miswrite_zipcodes
    )
    AGE_MISWRITING: ColumnNoiseType = ColumnNoiseType(
        "age_miswriting",
        noise_functions.miswrite_ages,
        token_noise_level=None,
        additional_parameters={"possible_perturbations": {-1: 0.5, 1: 0.5}},
    )
    NUMERIC_MISWRITING: ColumnNoiseType = ColumnNoiseType(
        "numeric_miswriting", noise_functions.miswrite_numerics
    )
    NICKNAME: ColumnNoiseType = ColumnNoiseType(
        "nickname", noise_functions.generate_nicknames, is_implemented=False,
    )  # TODO
    FAKE_NAME: ColumnNoiseType = ColumnNoiseType(
        "fake_names", noise_functions.generate_fake_names, is_implemented=False,
    )  # TODO
    PHONETIC: ColumnNoiseType = ColumnNoiseType(
        "phonetic", noise_functions.generate_phonetic_errors, is_implemented=False,
    )  # TODO
    OCR: ColumnNoiseType = ColumnNoiseType(
        "ocr", noise_functions.generate_ocr_errors, is_implemented=False,
    )  # TODO
    TYPOGRAPHIC: ColumnNoiseType = ColumnNoiseType(
        "typographic",
        noise_functions.generate_typographical_errors,
        additional_parameters={"include_original_token_level": 0.1},
    )


NOISE_TYPES = __NoiseTypes()
