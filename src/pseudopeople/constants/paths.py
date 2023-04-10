from pathlib import Path

import pseudopeople

BASE_DIR = Path(pseudopeople.__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"

INCORRECT_SELECT_NOISE_OPTIONS_DATA = DATA_ROOT / "incorrect_select_options.csv"
QWERTY_ERRORS = DATA_ROOT / "qwerty_errors.yaml"

SAMPLE_DATA_ROOT = DATA_ROOT / "sample_forms"
SAMPLE_DECENNIAL_CENSUS = SAMPLE_DATA_ROOT / "decennial_census_observer.parquet"
SAMPLE_TAXES_W2_AND_1099 = SAMPLE_DATA_ROOT / "tax_w2_observer.parquet"
SAMPLE_AMERICAN_COMMUNITIES_SURVEY = (
    SAMPLE_DATA_ROOT / "household_survey_observer_acs.parquet"
)
SAMPLE_CURRENT_POPULATION_SURVEY = SAMPLE_DATA_ROOT / "household_survey_observer_cps.parquet"
SAMPLE_SOCIAL_SECURITY = SAMPLE_DATA_ROOT / "social_security_observer.parquet"
SAMPLE_WOMEN_INFANTS_AND_CHILDREN = SAMPLE_DATA_ROOT / "wic_observer.parquet"
