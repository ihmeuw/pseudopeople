import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.noise_functions import (
    generate_fake_names,
    generate_missing_data,
    generate_nicknames,
    generate_phonetic_errors,
)
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
def default_configuration():
    return get_configuration()


def test_generate_missing_data(string_series, default_configuration):
    # TODO: [MIC-3910] Use custom config (MIC-3866)
    config = default_configuration["decennial_census"]["zipcode"]["missing_data"]
    noised_data = generate_missing_data(
        string_series, config, RANDOMNESS0, "test_missing_data"
    )
    noised_data_same_seed = generate_missing_data(
        string_series, config, RANDOMNESS0, "test_missing_data"
    )
    noised_data_different_seed = generate_missing_data(
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
