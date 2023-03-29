import random
import time
from string import ascii_lowercase, ascii_uppercase

import pandas as pd
import pytest

HOUSING_TYPES = [
    "Carceral",
    "College",
    "Military",
    "Nursing home",
    "Other institutional",
    "Other non-institutional",
    "Standard",
]

RACE_ETHNICITIES = [
    "AIAN",
    "Asian",
    "Black",
    "Latino",
    "Multiracial or Other",
    "NHOPI",
    "White",
]

RELATIONS_TO_HOUSEHOLD_HEAD = [
    "Adopted child",
    "Biological child",
    "Child-in-law",
    "Foster child",
    "Grandchild",
    "Institutionalized GQ pop",
    "Noninstitutionalized GQ pop",
    "Opp-sex partner",
    "Opp-sex spouse",
    "Other nonrelative",
    "Other relative",
    "Parent",
    "Parent-in-law",
    "Reference person",
    "Roommate",
    "Same-sex partner",
    "Same-sex spouse",
    "Sibling",
    "Stepchild",
]

DOB_START_DATE = time.mktime(time.strptime("1920-1-1", "%Y-%m-%d"))
DOB_END_DATE = time.mktime(time.strptime("2030-5-1", "%Y-%m-%d"))

STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DC",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


@pytest.fixture(scope="session")
def decennial_census_data_path(tmp_path_factory):
    """Generate a dummy decennial census dataframe, save to a tmpdir, and return that path."""
    random.seed(0)
    num_rows = 100_000
    data = pd.DataFrame(
        {
            "housing_type": [random.choice(HOUSING_TYPES) for _ in range(num_rows)],
            # TODO: Currently ages are actually floats but a followup pr will ensure ints
            "age": [
                str(random.randint(1, 100) + round(random.random(), 6))
                for _ in range(num_rows)
            ],
            "year": [random.choice(["2020", "2030"]) for _ in range(num_rows)],
            "race_ethnicity": [random.choice(RACE_ETHNICITIES) for _ in range(num_rows)],
            "guardian_1": [
                f"100_{random.randint(1,int(num_rows/3))}" for _ in range(num_rows)
            ],
            "first_name": [
                "First" + "".join(random.choice(ascii_lowercase) for _ in range(3))
                for _ in range(num_rows)
            ],
            "street_name": [
                "Street" + "".join(random.choice(ascii_lowercase) for _ in range(3))
                for _ in range(num_rows)
            ],
            "relation_to_household_head": [
                random.choice(RELATIONS_TO_HOUSEHOLD_HEAD) for _ in range(num_rows)
            ],
            # TODO: currently zipcodes are floats (and thus not zero-padded);
            # a followup PR will convert to 5-digit integer strings
            "zipcode": [str(random.randint(1, 99999)) + ".0" for _ in range(num_rows)],
            "date_of_birth": [
                time.strftime(
                    "%Y-%m-%d",
                    time.localtime(
                        DOB_START_DATE + random.random() * (DOB_END_DATE - DOB_START_DATE)
                    ),
                )
                for _ in range(num_rows)
            ],
            "simulant_id": ["100_" + str(i) for i in range(num_rows)],
            "middle_initial": [random.choice(ascii_uppercase) for _ in range(num_rows)],
            "city": [
                "City" + "".join(random.choice(ascii_lowercase) for _ in range(3))
                for _ in range(num_rows)
            ],
            "street_number": [str(random.randint(1, 15000)) for _ in range(num_rows)],
            "last_name": [
                "Last" + "".join(random.choice(ascii_lowercase) for _ in range(3))
                for _ in range(num_rows)
            ],
            "state": [random.choice(STATES) for _ in range(num_rows)],
            "sex": [random.choice(["Female", "Male"]) for _ in range(num_rows)],
            "unit_number": [
                "Unit " + "".join(random.choice(ascii_lowercase) for _ in range(3))
                for _ in range(num_rows)
            ],
            "guardian_2": [
                f"100_{random.randint(1,int(num_rows)/4)}" for _ in range(num_rows)
            ],
        }
    )

    data_path = tmp_path_factory.getbasetemp() / "dummy_data.csv"
    data.to_csv(data_path, index=False)

    return data_path
