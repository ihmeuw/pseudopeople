import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.utilities import get_configuration

RANDOMNESS0 = RandomnessStream(
    key="test_column_noise", clock=lambda: pd.Timestamp("2020-09-01"), seed=0
)
RANDOMNESS1 = RandomnessStream(
    key="test_column_noise", clock=lambda: pd.Timestamp("2020-09-01"), seed=1
)


@pytest.fixture(scope="module")
def string_series():
    num_simulants = 1_000_000
    return pd.Series([str(x) for x in range(num_simulants)])


@pytest.fixture(scope="module")
def categorical_series():
    return pd.Series(
        ["CA", "WA", "FL", "OR", "CO", "TX", "NY", "VA", "AZ", "''"] * 100_000, name="state"
    )


@pytest.fixture(scope="module")
def default_configuration():
    return get_configuration()


def test_generate_missing_data(string_series, default_configuration):
    # TODO: [MIC-3910] Use custom config (MIC-3866)
    config = default_configuration["decennial_census"]["zipcode"]["missing_data"]
    noised_data = NOISE_TYPES.MISSING_DATA(
        string_series, config, RANDOMNESS0, "test_missing_data"
    )
    noised_data_same_seed = NOISE_TYPES.MISSING_DATA(
        string_series, config, RANDOMNESS0, "test_missing_data"
    )
    noised_data_different_seed = NOISE_TYPES.MISSING_DATA(
        string_series, config, RANDOMNESS1, "test_missing_data"
    )

    # Confirm same randomness stream provides same results
    assert (noised_data == noised_data_same_seed).all()

    # Confirm different streams provide different results
    assert (noised_data != noised_data_different_seed).any()

    # Check for expected noise level
    expected_noise = config["row_noise_level"]
    actual_noise = (noised_data == "").mean()
    assert np.isclose(expected_noise, actual_noise, rtol=0.02)

    # Check that un-noised values are unchanged
    not_noised_idx = noised_data.index[noised_data != ""]
    assert "" not in noised_data[not_noised_idx].values
    assert (string_series[not_noised_idx] == noised_data[not_noised_idx]).all()


def test_incorrect_selection(categorical_series, default_configuration):
    config = default_configuration["decennial_census"]["state"]["incorrect_selection"]
    noised_data = NOISE_TYPES.INCORRECT_SELECTION(
        categorical_series, config, RANDOMNESS0, "test_incorrect_select"
    )
    noised_data_same_seed = NOISE_TYPES.INCORRECT_SELECTION(
        categorical_series, config, RANDOMNESS0, "test_incorrect_select"
    )
    noised_data_different_seed = NOISE_TYPES.INCORRECT_SELECTION(
        categorical_series, config, RANDOMNESS1, "test_incorrect_select"
    )

    # Confirm same randomness stream provides same results
    assert (noised_data == noised_data_same_seed).all()

    # Confirm different streams provide different results
    assert (noised_data != noised_data_different_seed).any()

    # Check for expected noise level
    expected_noise = config["row_noise_level"]
    # todo: Update when generate_incorrect_selection uses exclusive resampling
    # Get real expected noise to account for possibility of noising with original value
    # Here we have a a possibility of choosing any of the 50 states for our categorical series fixture
    expected_noise = expected_noise * (1 - 1 / 50)
    actual_noise = (noised_data != categorical_series).mean()
    assert np.isclose(expected_noise, actual_noise, rtol=0.02)

    original_empty_idx = categorical_series.index[categorical_series == ""]
    noised_empty_idx = noised_data.index[noised_data == ""]
    pd.testing.assert_index_equal(original_empty_idx, noised_empty_idx)


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
