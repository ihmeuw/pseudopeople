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

# TODO: Move into a metadata file and import metadata into prl
DATA_COLUMNS = ["year", "event_date", "survey_date", "tax_year"]


@pytest.mark.parametrize(
    "data_dir_name, noising_function, use_sample_data",
    [
        ("decennial_census_observer", generate_decennial_census, True),
        ("decennial_census_observer", generate_decennial_census, False),
        ("household_survey_observer_acs", generate_american_communities_survey, True),
        ("household_survey_observer_acs", generate_american_communities_survey, False),
        ("household_survey_observer_cps", generate_current_population_survey, True),
        ("household_survey_observer_cps", generate_current_population_survey, False),
        ("social_security_observer", generate_social_security, True),
        ("social_security_observer", generate_social_security, False),
        ("tax_w2_observer", generate_taxes_w2_and_1099, True),
        ("tax_w2_observer", generate_taxes_w2_and_1099, False),
        ("wic_observer", generate_women_infants_and_children, True),
        ("wic_observer", generate_women_infants_and_children, False),
        ("tax 1040", "todo", True),
        ("tax 1040", "todo", False),
    ],
)
def test_generate_form(
    data_dir_name: str, noising_function: Callable, use_sample_data: bool, tmpdir
):
    """Tests that noised forms are generated and as expected. The 'use_sample_data'
    parameter determines whether to use the sample data (if True) or
    a non-default root directory with multiple datasets to compile (if False)
    """
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement form {data_dir_name}")

    sample_data_path = list(
        (paths.SAMPLE_DATA_ROOT / data_dir_name).glob(f"{data_dir_name}*")
    )[0]

    # Load the unnoised sample data
    if sample_data_path.suffix == ".parquet":
        data = pd.read_parquet(sample_data_path)
    elif sample_data_path.suffix == ".hdf":
        data = pd.read_hdf(sample_data_path)
    else:
        raise NotImplementedError(
            f"Expected hdf or parquet but got {sample_data_path.suffix}"
        )

    # Configure if default (sample data) is used or a different root directory
    if use_sample_data:
        source = None  # will default to using sample data
    else:
        source = _generate_non_default_data_root(
            data_dir_name, tmpdir, sample_data_path, data
        )

    noised_data = noising_function(seed=0, source=source)
    noised_data_same_seed = noising_function(seed=0, source=source)
    noised_data_different_seed = noising_function(seed=1, source=source)

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


def _generate_non_default_data_root(data_dir_name, tmpdir, sample_data_path, data):
    """Helper function to break the single sample dataset into two and save
    out to tmpdir to be used as a non-default 'source' argument
    """
    outdir = tmpdir.mkdir(data_dir_name)
    suffix = sample_data_path.suffix
    split_idx = int(len(data) / 2)
    if suffix == ".parquet":
        data[:split_idx].to_parquet(outdir / f"{data_dir_name}_1{suffix}")
        data[split_idx:].to_parquet(outdir / f"{data_dir_name}_2{suffix}")
    elif suffix == ".hdf":
        data[:split_idx].to_hdf(
            outdir / f"{data_dir_name}_1{suffix}",
            "data",
            format="table",
            complib="bzip2",
            complevel=9,
            data_columns=DATA_COLUMNS,
        )
        data[split_idx:].to_hdf(
            outdir / f"{data_dir_name}_2{suffix}",
            "data",
            format="table",
            complib="bzip2",
            complevel=9,
            data_columns=DATA_COLUMNS,
        )
    else:
        raise NotImplementedError(f"Requires hdf or parquet, got {suffix}")
    return tmpdir


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
