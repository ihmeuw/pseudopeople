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
    config = get_configuration()[DATASETS.tax_w2_1099.name][Keys.ROW_NOISE][
        NOISE_TYPES.omission.name
    ]
    dataset_name_1 = "dummy_dataset_name"
    dataset_name_2 = DATASETS.tax_w2_1099.name
    noised_data1 = NOISE_TYPES.omission(dataset_name_1, dummy_data, config, RANDOMNESS)
    noised_data2 = NOISE_TYPES.omission(dataset_name_2, dummy_data, config, RANDOMNESS)

    expected_noise_1 = config[Keys.ROW_PROBABILITY]
    assert np.isclose(1 - len(noised_data1) / len(dummy_data), expected_noise_1, rtol=0.02)
    assert set(noised_data1.columns) == set(dummy_data.columns)
    assert (noised_data1.dtypes == dummy_data.dtypes).all()


def test_do_not_respond(mocker, dummy_data):
    config = get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.do_not_respond.name
    ]
    mocker.patch(
        "pseudopeople.noise_functions._get_census_do_not_respond_demographic_probabilities",
        side_effect=(lambda *_: config[Keys.ROW_PROBABILITY]),
    )
    dataset_name_1 = DATASETS.census.name
    dataset_name_2 = DATASETS.acs.name
    noised_data1 = NOISE_TYPES.do_not_respond(dataset_name_1, dummy_data, config, RANDOMNESS)
    noised_data2 = NOISE_TYPES.do_not_respond(dataset_name_2, dummy_data, config, RANDOMNESS)

    # Test that noising affects expected proportion with expected types
    assert np.isclose(
        1 - len(noised_data1) / len(dummy_data), config[Keys.ROW_PROBABILITY], rtol=0.02
    )
    assert set(noised_data1.columns) == set(dummy_data.columns)
    assert (noised_data1.dtypes == dummy_data.dtypes).all()

    # Check ACS data is scaled properly due to oversampling
    expected_noise = 0.5 + config[Keys.ROW_PROBABILITY] / 2
    assert np.isclose(1 - len(noised_data2) / len(dummy_data), expected_noise, rtol=0.02)
    assert set(noised_data2.columns) == set(dummy_data.columns)
    assert (noised_data2.dtypes == dummy_data.dtypes).all()
    assert len(noised_data1) != len(noised_data2)
    assert True


@pytest.mark.parametrize(
    "dataset_name",
    [ds.name for ds in DATASETS if ds not in [DATASETS.acs, DATASETS.cps, DATASETS.census]],
)
def test_do_not_respond_wrong_dataset(dataset_name, dummy_data):
    """Test do_not_respond only applies to expected datasets."""
    config = get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.do_not_respond.name
    ]
    with pytest.raises(ValueError, match=dataset_name):
        _ = NOISE_TYPES.do_not_respond(dataset_name, dummy_data, config, RANDOMNESS)


@pytest.mark.skip(reason="TODO")
def test_duplication():
    pass
