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

    omission: RowNoiseType = RowNoiseType("omission", noise_functions.omit_rows)
    # duplication: RowNoiseType = RowNoiseType("duplication", noise_functions.duplicate_rows,)
    missing_data: ColumnNoiseType = ColumnNoiseType(
        "missing_data",
        noise_functions.generate_missing_data,
        token_noise_level=None,
    )
    incorrect_selection: ColumnNoiseType = ColumnNoiseType(
        "incorrect_selection",
        noise_functions.generate_incorrect_selections,
        token_noise_level=None,
    )
    # copy_from_within_household: ColumnNoiseType = ColumnNoiseType(
    #     "copy_from_within_household",
    #     noise_functions.generate_within_household_copies,
    # )
    # month_day_swap: ColumnNoiseType = ColumnNoiseType(
    #     "month_day_swap",
    #     noise_functions.swap_months_and_days,
    # )
    zipcode_miswriting: ColumnNoiseType = ColumnNoiseType(
        "zipcode_miswriting",
        noise_functions.miswrite_zipcodes,
        token_noise_level=None,
        additional_parameters={
            "first_two_digits_noise_level": 0.04,
            "middle_digit_noise_level": 0.20,
            "last_two_digits_noise_level": 0.36,
        },
    )
    age_miswriting: ColumnNoiseType = ColumnNoiseType(
        "age_miswriting",
        noise_functions.miswrite_ages,
        token_noise_level=None,
        additional_parameters={"possible_perturbations": {-1: 0.5, 1: 0.5}},
    )
    numeric_miswriting: ColumnNoiseType = ColumnNoiseType(
        "numeric_miswriting",
        noise_functions.miswrite_numerics,
    )
    # nickname: ColumnNoiseType = ColumnNoiseType(
    #     "nickname",
    #     noise_functions.generate_nicknames,
    # )
    fake_name: ColumnNoiseType = ColumnNoiseType(
        "fake_name",
        noise_functions.generate_fake_names,
    )
    # phonetic: ColumnNoiseType = ColumnNoiseType(
    #     "phonetic",
    #     noise_functions.generate_phonetic_errors,
    # )
    # ocr: ColumnNoiseType = ColumnNoiseType(
    #     "ocr",
    #     noise_functions.generate_ocr_errors,
    # )
    typographic: ColumnNoiseType = ColumnNoiseType(
        "typographic",
        noise_functions.generate_typographical_errors,
        additional_parameters={"include_original_token_level": 0.1},
    )


NOISE_TYPES = __NoiseTypes()
