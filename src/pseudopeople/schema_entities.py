from enum import Enum
from typing import NamedTuple


# todo: is "form" the right word? Ask RT
class Form(Enum):
    """
    Enum containing all supported forms.
    """

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
