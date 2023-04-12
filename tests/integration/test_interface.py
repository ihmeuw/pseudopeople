from typing import Callable

import numpy as np
import pandas as pd
import pytest

from pseudopeople.constants import paths
from pseudopeople.interface import (
    generate_american_communities_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.schema_entities import COLUMNS, FORMS


@pytest.mark.parametrize(
    "data_dir_name, noising_function",
    [
        ("decennial_census_observer", generate_decennial_census),
        ("household_survey_observer_acs", generate_american_communities_survey),
        ("household_survey_observer_cps", generate_current_population_survey),
        ("social_security_observer", generate_social_security),
        ("tax_w2_observer", generate_taxes_w2_and_1099),
        ("wic_observer", generate_women_infants_and_children),
        ("tax 1040", "todo"),
    ],
)
def test_generate_form(data_dir_name: str, noising_function: Callable):
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement form {data_dir_name}")
    # todo fix hard-coding in MIC-3960
    data_path = paths.SAMPLE_DATA_ROOT / data_dir_name / f"{data_dir_name}.parquet"
    data = pd.read_parquet(data_path)

    noised_data = noising_function(seed=0)
    noised_data_same_seed = noising_function(seed=0)
    noised_data_different_seed = noising_function(seed=1)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)

    # Check each column's dtype
    for col in noised_data.columns:
        expected_dtype = [c.dtype_name for c in COLUMNS if c.name == col][0]
        if expected_dtype == np.dtype(str):
            # str dtype is 'object'
            expected_dtype = np.dtype(object)
        assert noised_data[col].dtype == expected_dtype


# TODO [MIC-4000]: add test that each col to get noised actually does get noised


@pytest.mark.parametrize(
    "data_dir_name, noising_function",
    [
        ("decennial_census_observer", generate_decennial_census),
        ("household_survey_observer_acs", generate_american_communities_survey),
        ("household_survey_observer_cps", generate_current_population_survey),
        ("social_security_observer", generate_social_security),
        ("tax_w2_observer", generate_taxes_w2_and_1099),
        ("wic_observer", generate_women_infants_and_children),
        ("tax 1040", "todo"),
    ],
)
def test_generate_form_with_year(data_dir_name: str, noising_function: Callable):
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement form {data_dir_name}")
    # todo fix hard-coding in MIC-3960
    data_path = paths.SAMPLE_DATA_ROOT / data_dir_name / f"{data_dir_name}.parquet"
    data = pd.read_parquet(data_path)

    noised_data = noising_function(year=2020, seed=0)
    noised_data_same_seed = noising_function(year=2020, seed=0)
    noised_data_different_seed = noising_function(year=2020, seed=1)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)


def _mock_extract_columns(columns_to_keep, noised_form):
    return noised_form


@pytest.mark.parametrize(
    "data_dir_name, noising_function, date_column",
    [
        ("decennial_census_observer", generate_decennial_census, FORMS.census.date_column),
        ("tax_w2_observer", generate_taxes_w2_and_1099, FORMS.tax_w2_1099.date_column),
        ("wic_observer", generate_women_infants_and_children, FORMS.wic.date_column),
        ("tax 1040", "todo", "todo"),
    ],
)
def test_form_filter_by_year(
    mocker, data_dir_name: str, noising_function: Callable, date_column: str
):
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement form {data_dir_name}")

    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_form", side_effect=_mock_noise_form)
    noised_data = noising_function(year=2020)

    assert (noised_data[date_column] == 2020).all()


def _mock_noise_form(
    form,
    form_data: pd.DataFrame,
    configuration,
    seed: int,
):
    """Mock noise_form that just returns unnoised data"""
    return form_data


@pytest.mark.parametrize(
    "data_dir_name, noising_function, form",
    [
        ("household_survey_observer_acs", generate_american_communities_survey, FORMS.acs),
        ("household_survey_observer_cps", generate_current_population_survey, FORMS.cps),
        ("social_security_observer", generate_social_security, FORMS.ssa),
    ],
)
def test_form_filter_by_year_with_full_dates(
    mocker, data_dir_name: str, noising_function: Callable, form: FORMS
):
    mocker.patch("pseudopeople.interface._extract_columns", side_effect=_mock_extract_columns)
    mocker.patch("pseudopeople.interface.noise_form", side_effect=_mock_noise_form)
    noised_data = noising_function(year=2020)

    dates = pd.DatetimeIndex(noised_data[form.date_column])
    if form == FORMS.ssa:
        assert (dates.year <= 2020).all()
    else:
        assert (dates.year == 2020).all()
