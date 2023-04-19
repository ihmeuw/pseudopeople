from dataclasses import dataclass
from typing import NamedTuple, Tuple

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES


class DtypeNames:
    """Container of expected dtype names"""

    CATEGORICAL = "category"
    DATETIME = "datetime64[ns]"
    OBJECT = "object"


@dataclass
class Column:
    name: str
    noise_types: Tuple[ColumnNoiseType, ...] = tuple()
    dtype_name: str = DtypeNames.OBJECT  # string dtype is 'object'


class __Columns(NamedTuple):
    """Container that contains information about columns that have potential to be noised"""

    age: Column = Column(
        "age",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.copy_from_within_household,
            NOISE_TYPES.age_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    city: Column = Column(
        "city",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    dob: Column = Column(
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
    employer_city: Column = Column(
        "employer_city",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_id: Column = Column(
        "employer_id",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_name: Column = Column(
        "employer_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_state: Column = Column(
        "employer_state",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        DtypeNames.CATEGORICAL,
    )
    employer_street_name: Column = Column(
        "employer_street_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_street_number: Column = Column(
        "employer_street_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_unit_number: Column = Column(
        "employer_unit_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    employer_zipcode: Column = Column(
        "employer_zipcode",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.zipcode_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    first_name: Column = Column(
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
    household_id: Column = Column(
        "household_id",
    )
    income: Column = Column(
        "income",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    itin: Column = Column(
        "itin",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.copy_from_within_household,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    last_name: Column = Column(
        "last_name",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.fake_name,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_city: Column = Column(
        "mailing_address_city",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_po_box: Column = Column(
        "mailing_address_po_box",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_state: Column = Column(
        "mailing_address_state",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        DtypeNames.CATEGORICAL,
    )
    mailing_street_name: Column = Column(
        "mailing_address_street_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_street_number: Column = Column(
        "mailing_address_street_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_unit_number: Column = Column(
        "mailing_address_unit_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    mailing_zipcode: Column = Column(
        "mailing_address_zipcode",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.zipcode_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    middle_initial: Column = Column(
        "middle_initial",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    race_ethnicity: Column = Column(
        "race_ethnicity",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        DtypeNames.CATEGORICAL,
    )
    relation_to_household_head: Column = Column(
        "relation_to_household_head",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        DtypeNames.CATEGORICAL,
    )
    sex: Column = Column(
        "sex",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        DtypeNames.CATEGORICAL,
    )
    simulant_id: Column = Column(
        "simulant_id",
    )
    ssa_event_date: Column = Column(
        "event_date",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.month_day_swap,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    ssa_event_type: Column = Column(
        "event_type",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        DtypeNames.CATEGORICAL,
    )
    ssn: Column = Column(
        "ssn",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.copy_from_within_household,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    state: Column = Column(
        "state",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        DtypeNames.CATEGORICAL,
    )
    street_name: Column = Column(
        "street_name",
        (
            NOISE_TYPES.missing_data,
            # NOISE_TYPES.phonetic,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    street_number: Column = Column(
        "street_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    survey_date: Column = Column(
        "survey_date",
        dtype_name=DtypeNames.DATETIME,
    )
    tax_form: Column = Column(
        "tax_form",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        DtypeNames.CATEGORICAL,
    )
    unit_number: Column = Column(
        "unit_number",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.numeric_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )
    zipcode: Column = Column(
        "zipcode",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.zipcode_miswriting,
            # NOISE_TYPES.ocr,
            NOISE_TYPES.typographic,
        ),
    )


COLUMNS = __Columns()


@dataclass
class Dataset:
    name: str
    columns: Tuple[Column, ...]  # This defines the output column order
    date_column: str

    row_noise_types: Tuple[RowNoiseType, ...] = (
        NOISE_TYPES.omission,
        # NOISE_TYPES.duplication,
    )


class __Datasets(NamedTuple):
    """NamedTuple that contains information about datasets and their related columns"""

    census: Dataset = Dataset(
        DatasetNames.CENSUS,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.age,
            COLUMNS.dob,
            COLUMNS.street_number,
            COLUMNS.street_name,
            COLUMNS.unit_number,
            COLUMNS.city,
            COLUMNS.state,
            COLUMNS.zipcode,
            COLUMNS.relation_to_household_head,
            COLUMNS.sex,
            COLUMNS.race_ethnicity,
        ),
        date_column="year",
    )
    acs: Dataset = Dataset(
        DatasetNames.ACS,
        columns=(  # This defines the output column order
            COLUMNS.household_id,
            COLUMNS.simulant_id,
            COLUMNS.survey_date,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.age,
            COLUMNS.dob,
            COLUMNS.street_number,
            COLUMNS.street_name,
            COLUMNS.unit_number,
            COLUMNS.city,
            COLUMNS.state,
            COLUMNS.zipcode,
            COLUMNS.sex,
            COLUMNS.race_ethnicity,
        ),
        date_column="survey_date",
    )
    cps: Dataset = Dataset(
        DatasetNames.CPS,
        columns=(  # This defines the output column order
            COLUMNS.household_id,
            COLUMNS.simulant_id,
            COLUMNS.survey_date,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.age,
            COLUMNS.dob,
            COLUMNS.street_number,
            COLUMNS.street_name,
            COLUMNS.unit_number,
            COLUMNS.city,
            COLUMNS.state,
            COLUMNS.zipcode,
            COLUMNS.sex,
            COLUMNS.race_ethnicity,
        ),
        date_column="survey_date",
    )
    wic: Dataset = Dataset(
        DatasetNames.WIC,
        columns=(  # This defines the output column order
            COLUMNS.household_id,
            COLUMNS.simulant_id,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.dob,
            COLUMNS.street_number,
            COLUMNS.street_name,
            COLUMNS.unit_number,
            COLUMNS.city,
            COLUMNS.state,
            COLUMNS.zipcode,
            COLUMNS.sex,
            COLUMNS.race_ethnicity,
        ),
        date_column="year",
    )
    ssa: Dataset = Dataset(
        DatasetNames.SSA,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.dob,
            COLUMNS.ssn,
            COLUMNS.ssa_event_type,
            COLUMNS.ssa_event_date,
        ),
        date_column="event_date",
    )
    tax_w2_1099: Dataset = Dataset(
        DatasetNames.TAXES_W2_1099,
        columns=(  # This defines the output column order
            COLUMNS.simulant_id,
            COLUMNS.first_name,
            COLUMNS.middle_initial,
            COLUMNS.last_name,
            COLUMNS.age,
            COLUMNS.dob,
            COLUMNS.mailing_street_number,
            COLUMNS.mailing_street_name,
            COLUMNS.mailing_unit_number,
            COLUMNS.mailing_po_box,
            COLUMNS.mailing_city,
            COLUMNS.mailing_state,
            COLUMNS.mailing_zipcode,
            COLUMNS.ssn,
            COLUMNS.income,
            COLUMNS.employer_id,
            COLUMNS.employer_name,
            COLUMNS.employer_street_number,
            COLUMNS.employer_street_name,
            COLUMNS.employer_unit_number,
            COLUMNS.employer_city,
            COLUMNS.employer_state,
            COLUMNS.employer_zipcode,
            COLUMNS.tax_form,
        ),
        date_column="tax_year",
    )
    # tax_1040: Dataset = Dataset(
    #     Datasets.TAXES_1040,
    # )


DATASETS = __Datasets()
