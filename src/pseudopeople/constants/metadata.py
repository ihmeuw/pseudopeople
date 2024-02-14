class DatasetNames:
    """Container for Dataset names"""

    ACS = "american_community_survey"
    CENSUS = "decennial_census"
    CPS = "current_population_survey"
    SSA = "social_security"
    TAXES_1040 = "taxes_1040"
    TAXES_W2_1099 = "taxes_w2_and_1099"
    WIC = "women_infants_and_children"


class __DateFormats:
    """Container that contains information about date formats"""

    YYYYMMDD = "%Y%m%d"
    MM_DD_YYYY = "%m/%d/%Y"
    MMDDYYYY = "%m%d%Y"


DATEFORMATS = __DateFormats()


# Value calculated for noise scaling for nicknames
# Constant calculated by number of names with nicknames / number of names used in PRL name mapping
# Found in PRL data.nickname_proportions.get_nickname_proportion()
NICKNAMES_PROPORTION = 0.5522180596459583


US_STATE_ABBRV_MAP = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC",
    # "PUERTO RICO": "PR",
}


YEAR_AGGREGATION_VALUE = 3000  # value for all years in a dataset for metadata proportions
