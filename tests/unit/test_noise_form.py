import random
from string import ascii_lowercase
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from vivarium.config_tree import ConfigTree

from pseudopeople.entity_types import ColumnNoiseType
from pseudopeople.interface import (
    generate_american_communities_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.noise import noise_form
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import Form


@pytest.fixture(scope="module")
def dummy_data():
    """Create a two-column dummy dataset"""
    random.seed(0)
    num_rows = 1_000_000
    return pd.DataFrame(
        {
            "numbers": [str(x) for x in range(num_rows)],
            "words": [
                "".join(random.choice(ascii_lowercase) for _ in range(4))
                for _ in range(num_rows)
            ],
        }
    )


@pytest.fixture(scope="module")
def dummy_config_noise_numbers():
    """Create a dummy configuration that applies all noise functions to a single
    column in the dummy_data fixture. All noise function specs are defined in
    reverse order here compared to how they are to be applied.

    NOTE: this is not a realistic scenario but allows for certain
    types of stress testing.
    """
    return ConfigTree(
        {
            "decennial_census": {
                "numbers": {
                    "missing_data": {"row_noise_level": 0.01},
                    "incorrect_selection": {"row_noise_level": 0.01},
                    "copy_from_within_household": {"row_noise_level": 0.01},
                    "month_day_swap": {"row_noise_level": 0.01},
                    "zipcode_miswriting": {
                        "row_noise_level": 0.01,
                        "zipcode_miswriting": [0.04, 0.04, 0.2, 0.36, 0.36],
                    },
                    "age_miswriting": {
                        "row_noise_level": 0.01,
                        "age_miswriting": [1, -1],
                    },
                    "numeric_miswriting": {
                        "row_noise_level": 0.01,
                        "numeric_miswriting": [0.1],
                    },
                    "nickname": {"row_noise_level": 0.01},
                    "fake_names": {"row_noise_level": 0.01},
                    "phonetic": {
                        "row_noise_level": 0.01,
                        "token_noise_level": 0.1,
                    },
                    "ocr": {
                        "row_noise_level": 0.01,
                        "token_noise_level": 0.1,
                    },
                    "typographic": {
                        "row_noise_level": 0.01,
                        "token_noise_level": 0.1,
                    },
                },
                "duplication": 0.01,
                "omission": 0.01,
            },
        }
    )


def test_noise_order(mocker, dummy_data, dummy_config_noise_numbers):
    """From docs: "Noising should be applied in the following order: omissions, duplications,
    missing data, incorrect selection, copy from w/in household, month and day
    swaps, zip code miswriting, age miswriting, numeric miswriting, nicknames,
    fake names, phonetic, OCR, typographic"
    """
    mock = mocker.MagicMock()
    # Mock the noise_functions functions so that they are not actually called and
    # return the original one-column dataframe (so that it doesn't become a mock
    # object itself after the first mocked function is applied.)
    mocker.patch(
        "pseudopeople.entity_types.get_index_to_noise", return_value=dummy_data.index
    )
    for field in NOISE_TYPES._fields:
        mock_return = (
            dummy_data[["numbers"]]
            if field in ["OMISSION", "DUPLICATION"]
            else dummy_data["numbers"]
        )
        mock.attach_mock(
            mocker.patch(
                f"pseudopeople.noise.NOISE_TYPES.{field}.noise_function",
                return_value=mock_return,
            ),
            field,
        )

    # FIXME: would be better to mock the form instead of using census
    noise_form(Form.CENSUS, dummy_data, dummy_config_noise_numbers, 0)

    call_order = [x[0] for x in mock.mock_calls if not x[0].startswith("__")]
    expected_call_order = [
        "OMISSION",
        "DUPLICATION",
        "MISSING_DATA",
        "INCORRECT_SELECTION",
        "COPY_FROM_WITHIN_HOUSEHOLD",
        "MONTH_DAY_SWAP",
        "ZIPCODE_MISWRITING",
        "AGE_MISWRITING",
        "NUMERIC_MISWRITING",
        "NICKNAME",
        "FAKE_NAME",
        "PHONETIC",
        "OCR",
        "TYPOGRAPHIC",
    ]

    assert expected_call_order == call_order


def test_columns_noised(dummy_data):
    """Test that the noise functions are only applied to the numbers column
    (as specified in the dummy config)
    """
    config = ConfigTree(
        {
            "decennial_census": {  # Does not really matter
                "numbers": {
                    "missing_data": {"row_noise_level": 0.1},
                },
            },
        },
    )
    noised_data = dummy_data.copy()
    noised_data = noise_form(Form.CENSUS, noised_data, config, 0)

    assert (dummy_data["numbers"] != noised_data["numbers"]).any()
    assert (dummy_data["words"] == noised_data["words"]).all()


@pytest.mark.parametrize(
    "func, form",
    [
        (generate_decennial_census, Form.CENSUS),
        (generate_american_communities_survey, Form.ACS),
        (generate_current_population_survey, Form.CPS),
        (generate_women_infants_and_children, Form.WIC),
        (generate_social_security, Form.SSA),
        (generate_taxes_w2_and_1099, Form.TAX_W2_1099),
        ("todo", Form.TAX_1040),
    ],
)
def test_correct_forms_are_used(func, form, mocker):
    """Test that each interface noise function uses the correct form"""
    if func == "todo":
        pytest.skip(reason=f"TODO: implement function for {form.value} form")
    mock = mocker.patch("pseudopeople.interface.noise_form")
    mocker.patch("pseudopeople.interface.pd.read_hdf", return_value=pd.DataFrame())
    _ = func("dummy/path.hdf")

    assert mock.call_args[0][0] == form


def test_two_noise_functions_are_independent(mocker):
    # Make simple config tree to test 2 noise functions work together
    config_tree = ConfigTree(
        {
            "decennial_census": {
                "fake_column_one": {
                    "alpha": {"row_noise_level": 0.20},
                    "beta": {"row_noise_level": 0.30},
                },
                "fake_column_two": {
                    "alpha": {"row_noise_level": 0.40},
                    "beta": {"row_noise_level": 0.50},
                },
            }
        }
    )

    # Mock objects for testing

    class MockNoiseTypes(NamedTuple):
        ALPHA: ColumnNoiseType = ColumnNoiseType(
            "alpha", lambda column, *_: column.str.cat(pd.Series("abc", index=column.index))
        )
        BETA: ColumnNoiseType = ColumnNoiseType(
            "beta", lambda column, *_: column.str.cat(pd.Series("123", index=column.index))
        )

    mock_noise_types = MockNoiseTypes()

    mocker.patch("pseudopeople.noise.NOISE_TYPES", mock_noise_types)
    dummy_form = pd.DataFrame(
        {
            "fake_column_one": ["cat", "dog", "bird", "bunny", "duck"] * 20_000,
            "fake_column_two": ["shoe", "pants", "shirt", "hat", "sunglasses"] * 20_000,
        }
    )

    noised_data = noise_form(
        form=Form.CENSUS,
        form_data=dummy_form,
        seed=0,
        configuration=config_tree,
    )

    # Get config values for testing
    col1_expected_abc_proportion = config_tree["decennial_census"]["fake_column_one"][
        "alpha"
    ]["row_noise_level"]
    col2_expected_abc_proportion = config_tree["decennial_census"]["fake_column_two"][
        "alpha"
    ]["row_noise_level"]
    col1_expected_123_proportion = config_tree["decennial_census"]["fake_column_one"]["beta"][
        "row_noise_level"
    ]
    col2_expected_123_proportion = config_tree["decennial_census"]["fake_column_two"]["beta"][
        "row_noise_level"
    ]

    assert np.isclose(
        noised_data["fake_column_one"].str.contains("abc").mean(),
        col1_expected_abc_proportion,
        rtol=0.01,
    )
    assert np.isclose(
        noised_data["fake_column_two"].str.contains("abc").mean(),
        col2_expected_abc_proportion,
        rtol=0.01,
    )
    assert np.isclose(
        noised_data["fake_column_one"].str.contains("123").mean(),
        col1_expected_123_proportion,
        rtol=0.01,
    )
    assert np.isclose(
        noised_data["fake_column_two"].str.contains("123").mean(),
        col2_expected_123_proportion,
        rtol=0.01,
    )

    # Assert columns experience both noise
    assert np.isclose(
        noised_data["fake_column_one"].str.contains("abc123").mean(),
        col1_expected_abc_proportion * col1_expected_123_proportion,
        rtol=0.01,
    )
    assert np.isclose(
        noised_data["fake_column_two"].str.contains("abc123").mean(),
        col2_expected_abc_proportion * col2_expected_123_proportion,
        rtol=0.01,
    )
    assert noised_data["fake_column_one"].str.contains("123abc").sum() == 0
    assert noised_data["fake_column_two"].str.contains("123abc").sum() == 0
