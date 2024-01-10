# Metadata container for noise types and column groupings
# Note we cannot import COLUMNS from schema entities because it creates a circular import

GUARDIAN_DUPLICATION_ADDRESS_COLUMNS = [
    "street_number",
    "street_name",
    "unit_number",
    "city",
    "state",
    "zipcode",
    "housing_type",
    "household_id",
    "year",
]


COPY_HOUSEHOLD_MEMBER_COLS = {
    "age": "copy_age",
    "date_of_birth": "copy_date_of_birth",
    "ssn": "copy_ssn",
    "spouse_ssn": "spouse_copy_ssn",
    "dependent_1_ssn": "dependent_1_copy_ssn",
    "dependent_2_ssn": "dependent_2_copy_ssn",
    "dependent_3_ssn": "dependent_3_copy_ssn",
    "dependent_4_ssn": "dependent_4_copy_ssn",
}


INT_COLUMNS = ["age", "wages", "mailing_po_box"]