class DatasetNames:
    """Container for Dataset names"""

    ACS = "american_community_survey"
    CENSUS = "decennial_census"
    CPS = "current_population_survey"
    SSA = "social_security"
    TAXES_1040 = "taxes_1040"
    TAXES_W2_1099 = "taxes_w2_and_1099"
    TAXES_DEPENDENTS = "taxes_dependents"
    WIC = "women_infants_and_children"


class __DateFormats:
    """Container that contains information about date formats"""

    YYYYMMDD = "%Y%m%d"
    MM_DD_YYYY = "%m/%d/%Y"


DATEFORMATS = __DateFormats()


class Attributes:
    DATE_FORMAT = "date_format"


# Value calculated for noise scaling for nicknames
# Constant calculated by number of names with nicknames / number of names used in PRL name mapping
# Found in PRL data.nickname_proportions.get_nickname_proportion()
NICKNAMES_PROPORTION = 0.5522180596459583
