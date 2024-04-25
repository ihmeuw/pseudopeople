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

# Columns that are integers in pseudopeople input but strings in
# the pseudopeople output. We have to be careful that they don't
# end up getting stringified as floats, e.g. age being 26.0 instead of 26
INT_TO_STRING_COLUMNS = ["age", "wages", "mailing_address_po_box"]


HOUSING_TYPE_GUARDIAN_DUPLICATION_RELATONSHIP_MAP = {
    "Carceral": "Institutionalized group quarters population",
    "Nursing home": "Institutionalized group quarters population",
    "Other institutional": "Institutionalized group quarters population",
    "College": "Noninstitutionalized group quarters population",
    "Military": "Noninstitutionalized group quarters population",
    "Other noninstitutional": "Noninstitutionalized group quarters population",
    "Household": "Other relative",
}
