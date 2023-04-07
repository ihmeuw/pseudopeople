from typing import NamedTuple, Tuple
from pseudopeople.noise_entities import NOISE_TYPES, RowNoiseType, ColumnNoiseType
from dataclasses import dataclass


@dataclass
class Column:
    name: str
    noise_types: Tuple[ColumnNoiseType]
    is_implemented: bool = True


class NoisedColumns:
    """Container that contains information about columns and their related
    noising functions"""

    AGE: Column = Column(
        "age",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.COPY_FROM_WITHIN_HOUSEHOLD,
            NOISE_TYPES.AGE_MISWRITING,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    CITY: Column = Column(
        "city",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.PHONETIC,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    DOB: Column = Column(
        "date_of_birth",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.COPY_FROM_WITHIN_HOUSEHOLD,
            NOISE_TYPES.MONTH_DAY_SWAP,
            NOISE_TYPES.NUMERIC_MISWRITING,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    EMPLOYER_CITY: str = "employer_city"
    EMPLOYER_ID: str = "employer_id"
    EMPLOYER_NAME: str = "employer_name"
    EMPLOYER_STATE: str = "employer_state"
    EMPLOYER_STREET_NAME: str = "employer_street_name"
    EMPLOYER_STREET_NUMBER: str = "employer_street_number"
    EMPLOYER_UNIT_NUMBER: str = "employer_unit_number"
    EMPLOYER_ZIPCODE: str = "employer_zipcode"
    FIRST_NAME: Column = Column(
        "first_name",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.NICKNAME,
            NOISE_TYPES.FAKE_NAME,
            NOISE_TYPES.PHONETIC,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    INCOME: str = "income"
    ITIN: str = "itin"
    LAST_NAME: Column = Column(
        "last_name",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.FAKE_NAME,
            NOISE_TYPES.PHONETIC,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    MAILING_CITY: str = "mailing_address_city"
    MAILING_PO_BOX: str = "mailing_address_po_box"
    MAILING_STATE: str = "mailing_address_state"
    MAILING_STREET_NAME: str = "mailing_address_street_name"
    MAILING_STREET_NUMBER: str = "mailing_address_street_number"
    MAILING_UNIT_NUMBER: str = "mailing_address_unit_number"
    MAILING_ZIPCODE: str = "mailing_address_zipcode"
    MIDDLE_INITIAL: Column = Column(
        "middle_initial",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.PHONETIC,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    RACE_ETHNICITY: Column = Column(
        "race_ethnicity",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.INCORRECT_SELECTION,
        ),
    )
    RELATION_TO_HOUSEHOLD_HEAD: Column = Column(
        "relation_to_household_head",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.INCORRECT_SELECTION,
        ),
    )
    SEX: Column = Column(
        "sex",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.INCORRECT_SELECTION,
        ),
    )
    SSA_EVENT_DATE: str = "event_date"
    SSA_EVENT_TYPE: str = "event_type"
    SSN: str = "ssn"
    STATE: Column = Column(
        "state",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.INCORRECT_SELECTION,
        ),
    )
    STREET_NAME: Column = Column(
        "street_name",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.PHONETIC,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    STREET_NUMBER: Column = Column(
        "street_number",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.NUMERIC_MISWRITING,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    TAX_FORM: str = "tax_form"
    UNIT_NUMBER: Column = Column(
        "unit_number",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.NUMERIC_MISWRITING,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )
    ZIPCODE: Column = Column(
        "zipcode",
        (
            NOISE_TYPES.MISSING_DATA,
            NOISE_TYPES.ZIPCODE_MISWRITING,
            NOISE_TYPES.OCR,
            NOISE_TYPES.TYPOGRAPHIC,
        ),
    )


@dataclass
class Form:
    name: str
    columns: Tuple[Column] = None
    row_noise_types: Tuple[RowNoiseType] = (NOISE_TYPES.OMISSION, NOISE_TYPES.DUPLICATION)
    is_implemented: bool = True


class __Forms(NamedTuple):
    """NamedTuple that contains information about forms and their related columns"""

    CENSUS: Form = Form(
        "decennial_census",
        columns=(
            NoisedColumns.FIRST_NAME,
            NoisedColumns.MIDDLE_INITIAL,
            NoisedColumns.LAST_NAME,
            NoisedColumns.AGE,
            NoisedColumns.DOB,
            NoisedColumns.STREET_NUMBER,
            NoisedColumns.STREET_NAME,
            NoisedColumns.UNIT_NUMBER,
            NoisedColumns.CITY,
            NoisedColumns.STATE,
            NoisedColumns.ZIPCODE,
            NoisedColumns.RELATION_TO_HOUSEHOLD_HEAD,
            NoisedColumns.SEX,
            NoisedColumns.RACE_ETHNICITY,
        ),
    )
    ACS: Form = Form(
        "american_communities_survey",
        is_implemented=False,
    )  # TODO
    CPS: Form = Form(
        "current_population_survey",
        is_implemented=False,
    )  # TODO
    WIC: Form = Form(
        "women_infants_and_children",
        is_implemented=False,
    )  # TODO
    SSA: Form = Form(
        "social_security",
        is_implemented=False,
    )  # TODO
    TAX_W2_1099: Form = Form(
        "taxes_w2_and_1099",
        is_implemented=False,
    )  # TODO
    TAX_1040: Form = Form(
        "taxes_1040",
        is_implemented=False,
    )  # TODO

FORMS = __Forms()