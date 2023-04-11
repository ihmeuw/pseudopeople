from typing import Callable

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
    assert set(noised_data.columns) == set(data.columns)
