import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("data", ["american_community_survey", "current_population_survey"])
def test_data_fixture_loading(data: str, output_dir: Path, request) -> None:
    """Test that fixtures are generated and resources are saved out"""
    start_time = time.time()
    df = request.getfixturevalue(data)
    end_time = time.time()
    # Getting the fixture should take some time to load
    generation_time = end_time - start_time
    assert generation_time > 0.5
    start_time = time.time()
    df2 = request.getfixturevalue(data)
    end_time = time.time()
    # Getting the fixture a second time should be fast
    assert np.isclose(end_time, start_time, rtol=0.0001)
    assert df.equals(df2)
    resources = pd.read_csv(output_dir / f"generate_{data}_resources.csv")


def test_data_fixture_year_default_filtering(data: str, request) -> None:
    pytest.main(["--year", "2020", "some_test_function"])
    df = request.getfixturevalue(data)

def test_data_fixture_year_custom_filtering(data: str, request) -> None:
    pass

def test_data_fixture_state_default_filtering(data: str, request) -> None:
    pass

def test_data_fixture_state_custom_filtering(data: str, request) -> None:
    pass