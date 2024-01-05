from pseudopeople.schema_entities import COLUMNS

GUARDIAN_DUPLICATION_ADDRESS_COLUMNS = [
    COLUMNS.street_number.name,
    COLUMNS.street_name.name,
    COLUMNS.unit_number.name,
    COLUMNS.city.name,
    COLUMNS.state.name,
    COLUMNS.zipcode.name,
    COLUMNS.housing_type.name,
    COLUMNS.household_id.name,
    COLUMNS.year.name,
]


COPY_HOUSEHOLD_MEMBER_COLS = {
    COLUMNS.age.name: COLUMNS.copy_age.name,
    COLUMNS.dob.name: COLUMNS.copy_date_of_birth.name,
    COLUMNS.ssn.name: COLUMNS.copy_ssn.name,
    COLUMNS.spouse_ssn.name: COLUMNS.spouse_copy_ssn.name,
    COLUMNS.dependent_1_ssn.name: COLUMNS.dependent_1_copy_ssn.name,
    COLUMNS.dependent_2_ssn.name: COLUMNS.dependent_2_copy_ssn.name,
    COLUMNS.dependent_3_ssn.name: COLUMNS.dependent_3_copy_ssn.name,
    COLUMNS.dependent_4_ssn.name: COLUMNS.dependent_4_copy_ssn.name,
}


INT_COLUMNS = [COLUMNS.age.name, COLUMNS.wages.name, COLUMNS.mailing_po_box.name]
