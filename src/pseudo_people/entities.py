from enum import Enum
from typing import NamedTuple

from pseudo_people import noise_functions
from pseudo_people.configuration import NoiseConfiguration
from pseudo_people.entity_types import ColumnNoiseType, RowNoiseType


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
    CITY: str = "city"
    # todo finish filling in columns


COLUMNS = __Columns()


class __NoiseTypes(NamedTuple):
    """
    Container for all noise types in the order in which they should be applied.
    """

    # todo finish filling in noise types in the correct order per the docs
    OMISSION: RowNoiseType = RowNoiseType("omission", noise_functions.omit_rows)
    DUPLICATION: RowNoiseType = RowNoiseType("duplication", noise_functions.duplicate_rows)
    NICKNAME: ColumnNoiseType = ColumnNoiseType(
        "nickname", noise_functions.generate_nicknames, {COLUMNS.FIRST_NAME}
    )
    FAKE_NAME: ColumnNoiseType = ColumnNoiseType(
        "fake_names",
        noise_functions.generate_fake_names,
        {COLUMNS.FIRST_NAME, COLUMNS.LAST_NAME},
    )
    PHONETIC: ColumnNoiseType = ColumnNoiseType(
        "phonetic",
        noise_functions.generate_phonetic_errors,
        {
            COLUMNS.FIRST_NAME,
            COLUMNS.MIDDLE_INITIAL,
            COLUMNS.LAST_NAME,
            COLUMNS.STREET_NAME,
            COLUMNS.CITY,
        },
    )


NOISE_TYPES = __NoiseTypes()


class ColumnNoiseParameter(Enum):
    """
    Enum containing all additional parameters used to specify column noise.
    """

    ROW_NOISE_LEVEL = "row_noise_level"
    TOKEN_NOISE_LEVEL = "token_noise_level"
    TYPOGRAPHIC_NOISE = "typographic_noise"
    AGE_MISWRITING = "age_miswriting"
    ZIPCODE_MISWRITING = "zipcode_miswriting"


# TODO create proper default configuration
DEFAULT_CONFIGURATION = NoiseConfiguration()
