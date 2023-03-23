import random
from string import ascii_lowercase

import pandas as pd
import pytest
from vivarium.config_tree import ConfigTree

from pseudopeople.entities import Form
from pseudopeople.interface import generate_decennial_census
from pseudopeople.noise import NOISE_TYPES, noise_form


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
    for field in NOISE_TYPES._fields:
        mock.attach_mock(
            mocker.patch(
                f"pseudopeople.noise.NOISE_TYPES.{field}.noise_function",
                return_value=dummy_data[["numbers"]],
            ),
            field,
        )
    # FIXME: would be better to mock the form instead of using census
    noise_form(Form.CENSUS, dummy_data, dummy_config_noise_numbers, 0)

    call_order = [call[0] for call in mock.mock_calls]
    expected_call_order = [
        "OMISSION",
        "DUPLICATION",
        "MISSING_DATA",
        "INCORRECT_SELECTION",
        "COPY_FROM_WITHIN_HOUSEHOLD",
        "MONTH_DAY_SWAP",
        "ZIP_CODE_MISWRITING",
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
        ("todo", Form.ACS),
        ("todo", Form.CPS),
        ("todo", Form.WIC),
        ("todo", Form.SSA),
        ("todo", Form.TAX_W2_1099),
        ("todo", Form.TAX_1040),
    ],
)
def test_correct_forms_are_used(func, form, mocker):
    """Test that each interface noise function uses the correct form"""
    if func == "todo":
        pytest.skip(reason=f"TODO: implement function for {form.value} form")
    mock = mocker.patch("pseudopeople.interface.noise_form")
    mocker.patch("pseudopeople.interface.pd")
    _ = func("dummy/path")

    assert mock.call_args[0][0] == form
