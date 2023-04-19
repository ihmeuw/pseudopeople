import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import DATASETS

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
    config = get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.omission.name
    ]
    dataset_name_1 = "dummy_dataset_name"
    dataset_name_2 = DATASETS.acs.name
    noised_data1 = NOISE_TYPES.omission(dataset_name_1, dummy_data, config, RANDOMNESS)
    noised_data2 = NOISE_TYPES.omission(dataset_name_2, dummy_data, config, RANDOMNESS)

    expected_noise_1 = config[Keys.PROBABILITY]
    assert np.isclose(1 - len(noised_data1) / len(dummy_data), expected_noise_1, rtol=0.02)
    assert set(noised_data1.columns) == set(dummy_data.columns)
    assert (noised_data1.dtypes == dummy_data.dtypes).all()

    # Check ACS data is scaled properly due to oversampling
    expected_noise_2 = 0.5 + config[Keys.PROBABILITY] / 2
    assert np.isclose(1 - len(noised_data2) / len(dummy_data), expected_noise_2, rtol=0.02)
    assert set(noised_data2.columns) == set(dummy_data.columns)
    assert (noised_data2.dtypes == dummy_data.dtypes).all()
    assert len(noised_data1) != len(noised_data2)


@pytest.mark.skip(reason="TODO")
def test_duplication():
    pass
