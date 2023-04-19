from typing import NamedTuple

from pseudopeople import noise_functions, utilities
from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType


class __NoiseTypes(NamedTuple):
    """Container for all noise types in the order in which they should be applied:
    omissions, duplications, missing data, incorrect selection, copy from w/in
    household, month and day swaps, zip code miswriting, age miswriting,
    numeric miswriting, nicknames, fake names, phonetic, OCR, typographic

    NOTE: Any configuration tree overwrites in these objects are what ends up
    in the "baseline" ConfigTree layer.
    """

    omission: RowNoiseType = RowNoiseType("omit_row", noise_functions.omit_rows)
    # duplication: RowNoiseType = RowNoiseType("duplicate_row", noise_functions.duplicate_rows)
    missing_data: ColumnNoiseType = ColumnNoiseType(
        "leave_blank",
        noise_functions.generate_missing_data,
    )
    incorrect_selection: ColumnNoiseType = ColumnNoiseType(
        "choose_wrong_option",
        noise_functions.generate_incorrect_selections,
        noise_level_scaling_function=utilities.noise_scaling_incorrect_selection,
    )
    # copy_from_within_household: ColumnNoiseType = ColumnNoiseType(
    #     "copy_from_household_member",
    #     noise_functions.generate_within_household_copies,
    # )
    # month_day_swap: ColumnNoiseType = ColumnNoiseType(
    #     "swap_month_and_day",
    #     noise_functions.swap_months_and_days,
    # )
    zipcode_miswriting: ColumnNoiseType = ColumnNoiseType(
        "write_wrong_zipcode_digits",
        noise_functions.miswrite_zipcodes,
        probability=None,
        additional_parameters={
            Keys.CELL_PROBABILITY: 0.01,
            Keys.ZIPCODE_DIGIT_PROBABILITIES: [0.04, 0.04, 0.20, 0.36, 0.36],
        },
    )
    age_miswriting: ColumnNoiseType = ColumnNoiseType(
        "misreport_age",
        noise_functions.miswrite_ages,
        additional_parameters={
            Keys.POSSIBLE_AGE_DIFFERENCES: {-2: 0.1, -1: 0.4, 1: 0.4, 2: 0.1}
        },
    )
    numeric_miswriting: ColumnNoiseType = ColumnNoiseType(
        "write_wrong_digits",
        noise_functions.miswrite_numerics,
        probability=None,
        additional_parameters={
            Keys.CELL_PROBABILITY: 0.01,
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )
    # nickname: ColumnNoiseType = ColumnNoiseType(
    #     "use_nickname",
    #     noise_functions.generate_nicknames,
    # )
    fake_name: ColumnNoiseType = ColumnNoiseType(
        "use_fake_name",
        noise_functions.generate_fake_names,
    )
    # phonetic: ColumnNoiseType = ColumnNoiseType(
    #     "make_phonetic_errors",
    #     noise_functions.generate_phonetic_errors,
    #     probability=None,
    #     additional_parameters={
    #         Keys.CELL_PROBABILITY: 0.01,
    #         Keys.TOKEN_PROBABILITY: 0.1,
    #     },
    # )
    # ocr: ColumnNoiseType = ColumnNoiseType(
    #     "make_ocr_errors",
    #     noise_functions.generate_ocr_errors,
    #     probability=None,
    #     additional_parameters={
    #         Keys.CELL_PROBABILITY: 0.01,
    #         Keys.TOKEN_PROBABILITY: 0.1,
    #     },
    # )
    typographic: ColumnNoiseType = ColumnNoiseType(
        "make_typos",
        noise_functions.generate_typographical_errors,
        probability=None,
        additional_parameters={  # TODO: need to clarify these
            Keys.CELL_PROBABILITY: 0.01,
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )


NOISE_TYPES = __NoiseTypes()
