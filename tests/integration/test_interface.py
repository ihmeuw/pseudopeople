from pathlib import Path
from typing import Callable, Union

import pandas as pd
import pytest

from pseudopeople.constants.paths import (
    SAMPLE_AMERICAN_COMMUNITIES_SURVEY,
    SAMPLE_CURRENT_POPULATION_SURVEY,
    SAMPLE_DECENNIAL_CENSUS,
    SAMPLE_SOCIAL_SECURITY,
    SAMPLE_TAXES_W2_AND_1099,
    SAMPLE_WOMEN_INFANTS_AND_CHILDREN,
)
from pseudopeople.interface import (
    generate_american_communities_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)


@pytest.mark.parametrize(
    "data_path, noising_function",
    [
        (SAMPLE_DECENNIAL_CENSUS, generate_decennial_census),
        (SAMPLE_AMERICAN_COMMUNITIES_SURVEY, generate_american_communities_survey),
        (SAMPLE_CURRENT_POPULATION_SURVEY, generate_current_population_survey),
        (SAMPLE_SOCIAL_SECURITY, generate_social_security),
        (SAMPLE_TAXES_W2_AND_1099, generate_taxes_w2_and_1099),
        (SAMPLE_WOMEN_INFANTS_AND_CHILDREN, generate_women_infants_and_children),
    ],
)
def test_generate_form(data_path: Union[Path, str], noising_function: Callable):
    data_path = Path(data_path)
    if data_path.suffix == ".hdf":
        data = pd.read_hdf(data_path)
    elif data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)

    noised_data = noising_function(source=data.copy(), seed=0)
    noised_data_same_seed = noising_function(source=data.copy(), seed=0)
    noised_data_different_seed = noising_function(source=data.copy(), seed=1)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)
    assert set(noised_data.columns) == set(data.columns)
