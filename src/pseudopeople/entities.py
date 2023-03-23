from enum import Enum
from typing import NamedTuple

from pseudopeople import noise_functions
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType


# todo: is "form" the right word? Ask RT
class Form(Enum):
    CENSUS = "decennial_census"
    ACS = "american_communities_survey"
    CPS = "current_population_survey"
    WIC = "women_infants_and_children"
    SSA = "social_security"
    TAX_W2_1099 = "taxes_w2_and_1099"
    TAX_1040 = "taxes_1040"


class __Columns(NamedTuple):
    FIRST_NAME: str = "first_name"
    MIDDLE_INITIAL: str = "middle_initial"
    LAST_NAME: str = "last_name"
    STREET_NAME: str = "street_name"
    ZIP_CODE: str = "zipcode"
    CITY: str = "city"
    AGE: str = "age"
    # todo finish filling in columns


COLUMNS = __Columns()


class __NoiseTypes(NamedTuple):
    """Container for all noise types in the order in which they should be applied:
    omissions, duplications, missing data, incorrect selection, copy from w/in
    household, month and day swaps, zip code miswriting, age miswriting,
    numeric miswriting, nicknames, fake names, phonetic, OCR, typographic"
    """

    # todo finish filling in noise types in the correct order per the docs
    OMISSION: RowNoiseType = RowNoiseType("omission", noise_functions.omit_rows)
    DUPLICATION: RowNoiseType = RowNoiseType("duplication", noise_functions.duplicate_rows)
    MISSING_DATA: ColumnNoiseType = ColumnNoiseType(
        "missing_data",
        noise_functions.generate_missing_data,
    )
    INCORRECT_SELECTION: ColumnNoiseType = ColumnNoiseType(
        "incorrect_selection", noise_functions.generate_incorrect_selections
    )
    COPY_FROM_WITHIN_HOUSEHOLD: ColumnNoiseType = ColumnNoiseType(
        "copy_from_within_household", noise_functions.generate_within_household_copies
    )
    MONTH_DAY_SWAP: ColumnNoiseType = ColumnNoiseType(
        "month_day_swap", noise_functions.swap_months_and_days
    )
    ZIP_CODE_MISWRITING: ColumnNoiseType = ColumnNoiseType(
        "zipcode_miswriting", noise_functions.miswrite_zip_codes
    )
    AGE_MISWRITING: ColumnNoiseType = ColumnNoiseType(
        "age_miswriting", noise_functions.miswrite_ages
    )
    NUMERIC_MISWRITING: ColumnNoiseType = ColumnNoiseType(
        "numeric_miswriting", noise_functions.miswrite_numerics
    )
    NICKNAME: ColumnNoiseType = ColumnNoiseType(
        "nickname", noise_functions.generate_nicknames
    )
    FAKE_NAME: ColumnNoiseType = ColumnNoiseType(
        "fake_names", noise_functions.generate_fake_names
    )
    PHONETIC: ColumnNoiseType = ColumnNoiseType(
        "phonetic", noise_functions.generate_phonetic_errors
    )
    OCR: ColumnNoiseType = ColumnNoiseType(
        # todo: implement the noise fn
        "ocr",
        noise_functions.generate_ocr_errors,
    )
    TYPOGRAPHIC: ColumnNoiseType = ColumnNoiseType(
        "typographic",
        noise_functions.generate_typographical_errors,
    )


NOISE_TYPES = __NoiseTypes()


class NoiseParameter(Enum):
    """
    Enum containing all additional parameters used to specify column noise.
    """

    ROW_NOISE_LEVEL = "row_noise_level"
    TOKEN_NOISE_LEVEL = "token_noise_level"
    TYPOGRAPHIC_NOISE = "typographic_noise"
    AGE_MISWRITING = "age_miswriting"
    ZIPCODE_MISWRITING = "zipcode_miswriting"
