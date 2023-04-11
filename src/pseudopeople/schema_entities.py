from dataclasses import dataclass
from typing import NamedTuple, Tuple

from pseudopeople.noise_entities import NOISE_TYPES, ColumnNoiseType, RowNoiseType


@dataclass
class NoisedColumn:
    name: str
    noise_types: Tuple[ColumnNoiseType, ...]


class NoisedColumns:
    """Container that contains information about columns and their related
    noising functions"""

    age: NoisedColumn = NoisedColumn(
        "age",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.copy_from_within_household,
            NOISE_TYPES.age_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    city: NoisedColumn = NoisedColumn(
        "city",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    dob: NoisedColumn = NoisedColumn(
        "date_of_birth",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.copy_from_within_household,
            # NOISE_TYPES.month_day_swap,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_city: NoisedColumn = NoisedColumn(
        "employer_city",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_id: NoisedColumn = NoisedColumn(
        "employer_id",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_name: NoisedColumn = NoisedColumn(
        "employer_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_state: NoisedColumn = NoisedColumn(
        "employer_state",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
    )
    employer_street_name: NoisedColumn = NoisedColumn(
        "employer_street_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_street_number: NoisedColumn = NoisedColumn(
        "employer_street_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_unit_number: NoisedColumn = NoisedColumn(
        "employer_unit_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_zipcode: NoisedColumn = NoisedColumn(
        "employer_zipcode",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.zipcode_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    first_name: NoisedColumn = NoisedColumn(
        "first_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.nickname,
            NOISE_TYPES.fake_name,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    income: NoisedColumn = NoisedColumn(
        "income",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    itin: NoisedColumn = NoisedColumn(
        "itin",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.copy_from_within_household,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    last_name: NoisedColumn = NoisedColumn(
        "last_name",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.fake_name,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_city: NoisedColumn = NoisedColumn(
        "mailing_address_city",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_po_box: NoisedColumn = NoisedColumn(
        "mailing_address_po_box",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_state: NoisedColumn = NoisedColumn(
        "mailing_address_state",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
    )
    mailing_street_name: NoisedColumn = NoisedColumn(
        "mailing_address_street_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_street_number: NoisedColumn = NoisedColumn(
        "mailing_address_street_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_unit_number: NoisedColumn = NoisedColumn(
        "mailing_address_unit_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_zipcode: NoisedColumn = NoisedColumn(
        "mailing_address_zipcode",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.zipcode_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    middle_initial: NoisedColumn = NoisedColumn(
        "middle_initial",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    race_ethnicity: NoisedColumn = NoisedColumn(
        "race_ethnicity",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
    )
    relation_to_household_head: NoisedColumn = NoisedColumn(
        "relation_to_household_head",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
    )
    sex: NoisedColumn = NoisedColumn(
        "sex",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
    )
    ssa_event_date: NoisedColumn = NoisedColumn(
        "event_date",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.month_day_swap,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    ssa_event_type: NoisedColumn = NoisedColumn(
        "event_type",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
    )
    ssn: NoisedColumn = NoisedColumn(
        "ssn",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.copy_from_within_household,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    state: NoisedColumn = NoisedColumn(
        "state",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
    )
    street_name: NoisedColumn = NoisedColumn(
        "street_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    street_number: NoisedColumn = NoisedColumn(
        "street_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    tax_form: NoisedColumn = NoisedColumn(
        "tax_form",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
    )
    unit_number: NoisedColumn = NoisedColumn(
        "unit_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    zipcode: NoisedColumn = NoisedColumn(
        "zipcode",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.zipcode_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )


@dataclass
class Form:
    name: str
    columns: Tuple[NoisedColumn, ...] = None
    row_noise_types: Tuple[RowNoiseType, ...] = (
        NOISE_TYPES.omission,
        # NOISE_TYPES.duplication,
    )


class __Forms(NamedTuple):
    """NamedTuple that contains information about forms and their related columns"""

    census: Form = Form(
        "decennial_census",
        columns=(
            NoisedColumns.first_name,
            NoisedColumns.middle_initial,
            NoisedColumns.last_name,
            NoisedColumns.age,
            NoisedColumns.dob,
            NoisedColumns.street_number,
            NoisedColumns.street_name,
            NoisedColumns.unit_number,
            NoisedColumns.city,
            NoisedColumns.state,
            NoisedColumns.zipcode,
            NoisedColumns.relation_to_household_head,
            NoisedColumns.sex,
            NoisedColumns.race_ethnicity,
        ),
    )
    acs: Form = Form(
        "american_communities_survey",
        columns=(
            NoisedColumns.first_name,
            NoisedColumns.middle_initial,
            NoisedColumns.last_name,
            NoisedColumns.age,
            NoisedColumns.dob,
            NoisedColumns.street_number,
            NoisedColumns.street_name,
            NoisedColumns.unit_number,
            NoisedColumns.city,
            NoisedColumns.state,
            NoisedColumns.zipcode,
            NoisedColumns.mailing_po_box,
            NoisedColumns.sex,
        ),
    )
    cps: Form = Form(
        "current_population_survey",
        columns=(
            NoisedColumns.first_name,
            NoisedColumns.middle_initial,
            NoisedColumns.last_name,
            NoisedColumns.age,
            NoisedColumns.dob,
            NoisedColumns.street_number,
            NoisedColumns.street_name,
            NoisedColumns.unit_number,
            NoisedColumns.mailing_po_box,
            NoisedColumns.city,
            NoisedColumns.state,
            NoisedColumns.zipcode,
            NoisedColumns.sex,
        ),
    )
    wic: Form = Form(
        "women_infants_and_children",
        columns=(
            NoisedColumns.first_name,
            NoisedColumns.middle_initial,
            NoisedColumns.last_name,
            NoisedColumns.age,
            NoisedColumns.dob,
            NoisedColumns.street_number,
            NoisedColumns.street_name,
            NoisedColumns.unit_number,
            NoisedColumns.mailing_po_box,
            NoisedColumns.city,
            NoisedColumns.state,
            NoisedColumns.zipcode,
            NoisedColumns.sex,
            NoisedColumns.race_ethnicity,
        ),
    )
    ssa: Form = Form(
        "social_security",
        columns=(
            NoisedColumns.first_name,
            NoisedColumns.middle_initial,
            NoisedColumns.last_name,
            NoisedColumns.age,
            NoisedColumns.dob,
            NoisedColumns.ssn,
            NoisedColumns.ssa_event_type,
            NoisedColumns.ssa_event_date,
        ),
    )
    tax_w2_1099: Form = Form(
        "taxes_w2_and_1099",
        columns=(
            NoisedColumns.first_name,
            NoisedColumns.middle_initial,
            NoisedColumns.last_name,
            NoisedColumns.age,
            NoisedColumns.dob,
            NoisedColumns.mailing_street_number,
            NoisedColumns.mailing_street_name,
            NoisedColumns.mailing_unit_number,
            NoisedColumns.mailing_city,
            NoisedColumns.mailing_state,
            NoisedColumns.mailing_po_box,
            NoisedColumns.mailing_zipcode,
            NoisedColumns.ssn,
            NoisedColumns.income,
            NoisedColumns.employer_id,
            NoisedColumns.employer_name,
            NoisedColumns.employer_street_number,
            NoisedColumns.employer_street_name,
            NoisedColumns.employer_unit_number,
            NoisedColumns.employer_city,
            NoisedColumns.employer_state,
            NoisedColumns.employer_zipcode,
            NoisedColumns.tax_form,
        ),
    )
    # tax_1040: Form = Form(
    #     "taxes_1040",
    # )


FORMS = __Forms()
