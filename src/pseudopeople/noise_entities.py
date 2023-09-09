from typing import NamedTuple

from pseudopeople import column_getters, noise_functions, noise_scaling
from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType


class __NoiseTypes(NamedTuple):
    """Container for all noise types in the order in which they should be applied:
    omit_row, do_not_respond, duplicate_row, leave_blank, choose_wrong_option,
    copy_from_household_member, swap_month_and_day, write_wrong_zipcode_digits,
    misreport_age, write_wrong_digits, use_nickname, use_fake_name,
    make_phonetic_errors, make_ocr_errors, make_typos

    NOTE: Any configuration tree overwrites in these objects are what ends up
    in the "baseline" ConfigTree layer.
    """

    omit_row: RowNoiseType = RowNoiseType("omit_row", noise_functions.omit_rows)
    do_not_respond: RowNoiseType = RowNoiseType(
        "do_not_respond", noise_functions.apply_do_not_respond
    )
    # duplicate_row: RowNoiseType = RowNoiseType("duplicate_row", noise_functions.duplicate_rows)
    leave_blank: ColumnNoiseType = ColumnNoiseType(
        "leave_blank",
        noise_functions.leave_blanks,
    )
    choose_wrong_option: ColumnNoiseType = ColumnNoiseType(
        "choose_wrong_option",
        noise_functions.choose_wrong_options,
        noise_level_scaling_function=noise_scaling.scale_choose_wrong_option,
    )
    copy_from_household_member: ColumnNoiseType = ColumnNoiseType(
        "copy_from_household_member",
        noise_functions.copy_from_household_member,
        noise_level_scaling_function=noise_scaling.scale_copy_from_household_member,
        additional_column_getter=column_getters.copy_from_household_member_column_getter,
    )
    swap_month_and_day: ColumnNoiseType = ColumnNoiseType(
        "swap_month_and_day",
        noise_functions.swap_months_and_days,
    )
    write_wrong_zipcode_digits: ColumnNoiseType = ColumnNoiseType(
        "write_wrong_zipcode_digits",
        noise_functions.write_wrong_zipcode_digits,
        additional_parameters={
            Keys.ZIPCODE_DIGIT_PROBABILITIES: [0.04, 0.04, 0.20, 0.36, 0.36],
        },
    )
    misreport_age: ColumnNoiseType = ColumnNoiseType(
        "misreport_age",
        noise_functions.misreport_ages,
        additional_parameters={
            Keys.POSSIBLE_AGE_DIFFERENCES: {-2: 0.1, -1: 0.4, 1: 0.4, 2: 0.1}
        },
    )
    write_wrong_digits: ColumnNoiseType = ColumnNoiseType(
        "write_wrong_digits",
        noise_functions.write_wrong_digits,
        additional_parameters={
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )
    use_nickname: ColumnNoiseType = ColumnNoiseType(
        "use_nickname",
        noise_functions.use_nicknames,
        noise_level_scaling_function=noise_scaling.scale_nicknames,
    )
    use_fake_name: ColumnNoiseType = ColumnNoiseType(
        "use_fake_name",
        noise_functions.use_fake_names,
    )
    make_phonetic_errors: ColumnNoiseType = ColumnNoiseType(
        "make_phonetic_errors",
        noise_functions.make_phonetic_errors,
        additional_parameters={
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )
    make_ocr_errors: ColumnNoiseType = ColumnNoiseType(
        "make_ocr_errors",
        noise_functions.make_ocr_errors,
        additional_parameters={
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )
    make_typos: ColumnNoiseType = ColumnNoiseType(
        "make_typos",
        noise_functions.make_typos,
        additional_parameters={
            Keys.TOKEN_PROBABILITY: 0.1,
        },
    )


NOISE_TYPES = __NoiseTypes()
