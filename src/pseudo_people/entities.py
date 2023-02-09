from enum import Enum
from typing import NamedTuple

from pseudo_people import noise_functions
from pseudo_people.entity_types import ColumnMetadata, ColumnNoiseType, RowNoiseType


class Form(Enum):
    CENSUS = "decennial_census"
    ACS = "american_communities_survey"
    CPS = "current_population_survey"
    WIC = "women_infants_and_children"
    SSA = "social_security"
    TAX_W2_1099 = "taxes_w2_and_1099"
    TAX_1040 = "taxes_1040"


class __RowNoiseTypes(NamedTuple):
    OMISSION = RowNoiseType("omission", noise_functions.omit_rows)
    DUPLICATION = RowNoiseType("duplication", noise_functions.duplicate_rows)


ROW_NOISE_TYPES = __RowNoiseTypes()


class __ColumnNoiseTypes(NamedTuple):
    """
    Container for all column noise types.
    """

    NICKNAME = ColumnNoiseType("nickname", noise_functions.generate_nicknames)
    FAKE_NAME = ColumnNoiseType("fake_names", noise_functions.generate_fake_names)
    PHONETIC = ColumnNoiseType("phonetic", noise_functions.generate_phonetic_errors)
    # todo finish filling in noise types


COLUMN_NOISE_TYPES = __ColumnNoiseTypes()


class ColumnNoiseParameter(Enum):
    """
    Enum containing all additional parameters used to specify column noise.
    """

    ROW_NOISE_LEVEL = "row_noise_level"
    TOKEN_NOISE_LEVEL = "token_noise_level"
    TYPOGRAPHIC_NOISE = "typographic_noise"
    AGE_MISWRITING = "age_miswriting"
    ZIPCODE_MISWRITING = "zipcode_miswriting"


class __ColumnsMetadata(NamedTuple):
    """
    Container for the metadata required to add noise for all possible columns
    """

    FIRST_NAME = ColumnMetadata(
        "first_name",
        [
            COLUMN_NOISE_TYPES.NICKNAME,
            COLUMN_NOISE_TYPES.FAKE_NAME,
            COLUMN_NOISE_TYPES.PHONETIC,
            # todo add remaining column noise
        ],
    )
    MIDDLE_INITIAL = ColumnMetadata("middle_initial", [])
    # todo finish filling in columns

    def __getitem__(self, column_name: str) -> ColumnMetadata:
        # todo
        ...


COLUMNS_METADATA = __ColumnsMetadata()
