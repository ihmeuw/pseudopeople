import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.configuration import get_configuration
from pseudopeople.noise_entities import NOISE_TYPES

RANDOMNESS = RandomnessStream(
    key="test_row_noise", clock=lambda: pd.Timestamp("2020-09-01"), seed=0
)


@pytest.fixture()
def dummy_data():
    num_simulants = 1_000_000
    dummy_idx = pd.Index(range(num_simulants))

    numbers_list = list(range(10))
    numbers_series = pd.Series(numbers_list * int(num_simulants / len(numbers_list)))
    word_list = ["word_1", "word_2", "word_3", "word_4", "word5"]
    word_series = pd.Series(word_list * int(num_simulants / len(word_list)))

    return pd.DataFrame(
        {
            "numbers": numbers_series,
            "words": word_series,
        },
        index=dummy_idx,
    )


def test_omission(dummy_data):
    config = get_configuration().decennial_census.row_noise.omission
    noised_data = NOISE_TYPES.omission(dummy_data, config, RANDOMNESS)

    expected_noise = config.probability
    assert np.isclose(1 - len(noised_data) / len(dummy_data), expected_noise, rtol=0.02)
    assert set(noised_data.columns) == set(dummy_data.columns)
    assert (noised_data.dtypes == dummy_data.dtypes).all()


@pytest.mark.skip(reason="TODO")
def test_duplication():
    pass
