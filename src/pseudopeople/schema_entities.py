from dataclasses import dataclass
from typing import NamedTuple, Tuple

from pseudopeople.noise_entities import NOISE_TYPES, ColumnNoiseType, RowNoiseType


@dataclass
class NoisedColumn:
    name: str
    noise_types: Tuple[ColumnNoiseType]
    is_implemented: bool = True


class NoisedColumns:
    """Container that contains information about columns and their related
    noising functions"""

    age: NoisedColumn = NoisedColumn(
        "age",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.copy_from_within_household,
            NOISE_TYPES.age_miswriting,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    city: NoisedColumn = NoisedColumn(
        "city",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.phonetic,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    dob: NoisedColumn = NoisedColumn(
        "date_of_birth",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.copy_from_within_household,
            NOISE_TYPES.month_day_swap,
            NOISE_TYPES.numeric_miswriting,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_city: str = "employer_city"
    employer_id: str = "employer_id"
    employer_name: str = "employer_name"
    employer_state: str = "employer_state"
    employer_street_name: str = "employer_street_name"
    employer_street_number: str = "employer_street_number"
    employer_unit_number: str = "employer_unit_number"
    employer_zipcode: str = "employer_zipcode"
    first_name: NoisedColumn = NoisedColumn(
        "first_name",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.nickname,
            NOISE_TYPES.fake_name,
            NOISE_TYPES.phonetic,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    income: str = "income"
    itin: str = "itin"
    last_name: NoisedColumn = NoisedColumn(
        "last_name",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.fake_name,
            NOISE_TYPES.phonetic,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_city: str = "mailing_address_city"
    mailing_po_box: str = "mailing_address_po_box"
    mailing_state: str = "mailing_address_state"
    mailing_street_name: str = "mailing_address_street_name"
    mailing_street_number: str = "mailing_address_street_number"
    mailing_unit_number: str = "mailing_address_unit_number"
    mailing_zipcode: str = "mailing_address_zipcode"
    middle_initial: NoisedColumn = NoisedColumn(
        "middle_initial",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.phonetic,
            NOISE_TYPES.ocr,
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
    ssa_event_date: str = "event_date"
    ssa_event_type: str = "event_type"
    ssn: str = "ssn"
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
            NOISE_TYPES.phonetic,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    street_number: NoisedColumn = NoisedColumn(
        "street_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    tax_form: str = "tax_form"
    unit_number: NoisedColumn = NoisedColumn(
        "unit_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    zipcode: NoisedColumn = NoisedColumn(
        "zipcode",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.zipcode_miswriting,
            NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )


@dataclass
class Form:
    name: str
    columns: Tuple[NoisedColumn] = None
    row_noise_types: Tuple[RowNoiseType] = (NOISE_TYPES.omission, NOISE_TYPES.duplication)
    is_implemented: bool = True


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
        is_implemented=False,
    )  # TODO
    cps: Form = Form(
        "current_population_survey",
        is_implemented=False,
    )  # TODO
    wic: Form = Form(
        "women_infants_and_children",
        is_implemented=False,
    )  # TODO
    ssa: Form = Form(
        "social_security",
        is_implemented=False,
    )  # TODO
    tax_w2_1099: Form = Form(
        "taxes_w2_and_1099",
        is_implemented=False,
    )  # TODO
    tax_1040: Form = Form(
        "taxes_1040",
        is_implemented=False,
    )  # TODO


FORMS = __Forms()
