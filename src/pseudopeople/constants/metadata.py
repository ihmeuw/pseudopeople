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
