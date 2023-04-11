from dataclasses import dataclass
from typing import NamedTuple, Tuple
import numpy as np
import pandas as pd

from pseudopeople.constants import metadata
from pseudopeople.noise_entities import NOISE_TYPES, ColumnNoiseType, RowNoiseType


@dataclass
class Column:
    name: str
    noise_types: Tuple[ColumnNoiseType, ...] = tuple()
    dtype: np.dtype = str


class Columns:
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
        pd.CategoricalDtype(categories=metadata.STATES),
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
        pd.CategoricalDtype(categories=metadata.STATES),
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
        pd.CategoricalDtype(categories=metadata.RACE_ETHNICITIES),
    )
    relation_to_household_head: Column = Column(
        "relation_to_household_head",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        pd.CategoricalDtype(categories=metadata.RELATIONSHIPS),
    )
    sex: Column = Column(
        "sex",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        pd.CategoricalDtype(categories=metadata.SEXES),
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
        pd.CategoricalDtype(categories=metadata.SSA_EVENT_TYPES),
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
        pd.CategoricalDtype(categories=metadata.STATES),
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
        dtype="datetime64[ns]",
    )
    tax_form: Column = Column(
        "tax_form",
        (
            NOISE_TYPES.missing_data,
            NOISE_TYPES.incorrect_selection,
        ),
        pd.CategoricalDtype(categories=metadata.TAX_FORMS),
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


@dataclass
class Form:
    name: str
    columns: Tuple[Column, ...]  # This defines the output column order
    row_noise_types: Tuple[RowNoiseType, ...] = (
        NOISE_TYPES.omission,
        # NOISE_TYPES.duplication,
    )


class __Forms(NamedTuple):
    """NamedTuple that contains information about forms and their related columns"""

    census: Form = Form(
        "decennial_census",
        columns=(  # This defines the output column order
            Columns.simulant_id,
            Columns.first_name,
            Columns.middle_initial,
            Columns.last_name,
            Columns.age,
            Columns.dob,
            Columns.street_number,
            Columns.street_name,
            Columns.unit_number,
            Columns.city,
            Columns.state,
            Columns.zipcode,
            Columns.relation_to_household_head,
            Columns.sex,
            Columns.race_ethnicity,
        ),
    )
    acs: Form = Form(
        "american_communities_survey",
        columns=(  # This defines the output column order
            Columns.household_id,
            Columns.simulant_id,
            Columns.survey_date,
            Columns.first_name,
            Columns.middle_initial,
            Columns.last_name,
            Columns.age,
            Columns.dob,
            Columns.street_number,
            Columns.street_name,
            Columns.unit_number,
            Columns.city,
            Columns.state,
            Columns.zipcode,
            Columns.sex,
            Columns.race_ethnicity,
        ),
    )
    cps: Form = Form(
        "current_population_survey",
        columns=(  # This defines the output column order
            Columns.household_id,
            Columns.simulant_id,
            Columns.survey_date,
            Columns.first_name,
            Columns.middle_initial,
            Columns.last_name,
            Columns.age,
            Columns.dob,
            Columns.street_number,
            Columns.street_name,
            Columns.unit_number,
            Columns.city,
            Columns.state,
            Columns.zipcode,
            Columns.sex,
            Columns.race_ethnicity,
        ),
    )
    wic: Form = Form(
        "women_infants_and_children",
        columns=(  # This defines the output column order
            Columns.household_id,
            Columns.simulant_id,
            Columns.first_name,
            Columns.middle_initial,
            Columns.last_name,
            Columns.dob,
            Columns.street_number,
            Columns.street_name,
            Columns.unit_number,
            Columns.city,
            Columns.state,
            Columns.zipcode,
            Columns.sex,
            Columns.race_ethnicity,
        ),
    )
    ssa: Form = Form(
        "social_security",
        columns=(  # This defines the output column order
            Columns.simulant_id,
            Columns.first_name,
            Columns.middle_initial,
            Columns.last_name,
            Columns.dob,
            Columns.ssn,
            Columns.ssa_event_type,
            Columns.ssa_event_date,
        ),
    )
    tax_w2_1099: Form = Form(
        "taxes_w2_and_1099",
        columns=(  # This defines the output column order
            Columns.simulant_id,
            Columns.first_name,
            Columns.middle_initial,
            Columns.last_name,
            Columns.age,
            Columns.dob,
            Columns.mailing_street_number,
            Columns.mailing_street_name,
            Columns.mailing_unit_number,
            Columns.mailing_po_box,
            Columns.mailing_city,
            Columns.mailing_state,
            Columns.mailing_zipcode,
            Columns.ssn,
            Columns.income,
            Columns.employer_id,
            Columns.employer_name,
            Columns.employer_street_number,
            Columns.employer_street_name,
            Columns.employer_unit_number,
            Columns.employer_city,
            Columns.employer_state,
            Columns.employer_zipcode,
            Columns.tax_form,
        ),
    )
    # tax_1040: Form = Form(
    #     "taxes_1040",
    # )


FORMS = __Forms()
