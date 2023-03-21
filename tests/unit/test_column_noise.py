import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.noise_functions import (
    generate_fake_names,
    generate_nicknames,
    generate_phonetic_errors,
    missing_data,
)
from pseudopeople.utilities import get_configuration

RANDOMNESS = RandomnessStream(
    key="test_column_noise", clock=lambda: pd.Timestamp("2020-09-01"), seed=0
)


@pytest.fixture(scope="module")
def string_series():
    num_simulants = 1_000_000
    return pd.Series([str(x) for x in range(num_simulants)])


@pytest.fixture(scope="module")
def default_configuration():
    return get_configuration()


def test_missing_data(string_series, default_configuration):
    # TODO: [MIC-3910] Use custom config (MIC-3866)
    config = default_configuration["decennial_census"]["zipcode"]["missing_data"]
    noised_data = missing_data(string_series, config, RANDOMNESS, "test_missing_data")
    expected_noise = config["row_noise_level"]
    actual_noise = (noised_data == "").mean()

    assert np.isclose(expected_noise, actual_noise, rtol=0.02)


@pytest.mark.skip(reason="TODO")
def test_incorrect_selection():
    pass


@pytest.mark.skip(reason="TODO")
def test_copy_from_within_household():
    pass


@pytest.mark.skip(reason="TODO")
def test_swap_month_day():
    pass


@pytest.mark.skip(reason="TODO")
def test_miswrite_zipcode():
    pass


@pytest.mark.skip(reason="TODO")
def test_miswrite_age():
    pass


@pytest.mark.skip(reason="TODO")
def test_miswrite_numeric():
    pass


@pytest.mark.skip(reason="TODO")
def test_nickname_noise():
    pass


@pytest.mark.skip(reason="TODO")
def test_fake_name_noise():
    pass


@pytest.mark.skip(reason="TODO")
def test_phonetic_noise():
    pass


@pytest.mark.skip(reason="TODO")
def test_ocr_noise():
    pass


@pytest.mark.skip(reason="TODO")
def test_typographic_noise():
    pass
