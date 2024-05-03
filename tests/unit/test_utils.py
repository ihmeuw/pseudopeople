import numpy as np
import pandas as pd
import pytest
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap

from pseudopeople.dtypes import DtypeNames
from pseudopeople.noise_functions import _corrupt_tokens
from pseudopeople.utilities import (
    get_index_to_noise,
    to_string_as_integer,
    two_d_array_choice,
    vectorized_choice,
)
from tests.conftest import FuzzyChecker

RANDOMNESS0 = RandomnessStream(
    key="test_utils",
    clock=lambda: pd.Timestamp("2020-09-01"),
    seed=0,
    index_map=IndexMap(),
)
CORRUPT_TOKENS_TEST_CASES = {
    # Possible tokens to noise: abc, ab, a, c
    # Tuples of (token noised, token not noised)
    # Example: "abc" can be noised to "heybsee". This means "a" becomes "hey"
    # and "c" becomes "see" which means 2 tokens are noised ("a" and "c") and 2
    # are not ("abc" and "ab").
    # If a overlapping token is noised, we proceed with assuming the shorter
    # token didn't exist.
    "abc": {
        "abc": (0, 4),
        "absee": (1, 3),
        "tyc": (1, 2),
        "heybsee": (2, 2),
        "heybc": (1, 3),
        "tysee": (2, 1),
        "def": (1, 0),
    },
    "abzc": {
        # No "abc" token and z is not a token we noise
        "abzc": (0, 3),
        "abzsee": (1, 2),
        "tyzc": (1, 1),
        "heybzsee": (2, 1),
        "tyzsee": (2, 0),
        "heybzc": (1, 2),
    },
    "jkl": {
        # No tokens - to test nothing is noised
        "jkl": (0, 0),
    },
}


def test_to_string_as_integer():
    # This tests that object columns return only strings.
    # This is to handle dtype issues we were having with int/float/strings in
    # age, wages, and po box columns.
    s = pd.Series([np.nan, 1, "2.0", 3.01, 4.0, "5.055", np.nan])
    t = to_string_as_integer(s)
    assert s.dtype.name == t.dtype.name
    assert t.dtype.name == DtypeNames.OBJECT

    expected = pd.Series([np.nan, "1", "2", "3", "4", "5", np.nan])

    pd.testing.assert_series_equal(t, expected)


@pytest.mark.parametrize("pair", CORRUPT_TOKENS_TEST_CASES.items())
def test__corrupt_tokens(pair, fuzzy_checker: FuzzyChecker):
    """
    Unit test for _corrupt_tokens. We want to test that the noise level is correct, that the
    correct tokens are noised, and that the correct behavior happens (meaning the longer tokens
    are noised first) in a string).
    """
    fake_errors = pd.DataFrame(
        {
            "options": [
                "def",
                "ty",
                "hey",
                "see",
            ],
        },
        index=["abc", "ab", "a", "c"],
    )
    string, pathways = pair

    data = pd.Series([string] * 100_000, name="column")
    token_probability = 0.4

    noised = _corrupt_tokens(
        errors=fake_errors,
        column=data,
        token_probability=token_probability,
        randomness_stream=RANDOMNESS0,
        addl_key="test__corrupt_tokens",
    )

    # Assert our noised data is one of our possible strings. This also checks that
    # the tokens are noised in the correct order (longer tokens first) because if
    # a longer token overlaps a shorter token and the longer token is noised, we
    # do not count the shorter token as a possiblility of being noised.
    assert noised.isin(pathways.keys()).all()
    for result, (num_noised, num_not_noised) in pathways.items():
        pathway_probability = (
            token_probability**num_noised * (1 - token_probability) ** num_not_noised
        )
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_phonetic_error_values",
            observed_numerator=(noised == result).sum(),
            observed_denominator=len(data),
            target_proportion=pathway_probability,
            name_additional=f"result {result} from string {string}",
        )

    # Assert token noise level is correct.
    if string == "abc":
        tokens_per_string = 4
    elif string == "abzc":
        tokens_per_string = 3
    elif string == "jkl":
        tokens_per_string = 0

    any_token_noised = 1 - (1 - token_probability) ** tokens_per_string
    fuzzy_checker.fuzzy_assert_proportion(
        name="generate_phonetic_errors",
        observed_numerator=(noised != data).sum(),
        observed_denominator=len(data),
        target_proportion=any_token_noised,
    )


def test__corrupt_tokens_multiple_options(fuzzy_checker: FuzzyChecker):
    """
    Tests that multiple options can be chosen for a token
    """
    fake_errors = pd.DataFrame(
        {
            "option1": ["def"],
            "option2": ["ghi"],
            "option3": ["jkl"],
        },
        index=["abc"],
    )
    data = pd.Series(["abc"] * 100_000, name="column")
    token_probability = 0.4

    noised = _corrupt_tokens(
        errors=fake_errors,
        column=data,
        token_probability=token_probability,
        randomness_stream=RANDOMNESS0,
        addl_key="test__corrupt_tokens",
    )
    strings = ["abc", "def", "ghi", "jkl"]
    assert (noised.isin(strings)).all()

    for string in strings:
        proportion = 1 - token_probability if string == "abc" else token_probability / 3
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_multiple_options__corruprt_tokens",
            observed_numerator=(noised == string).sum(),
            observed_denominator=len(data),
            target_proportion=proportion,
            name_additional=f"result {string}",
        )


def test_get_index_to_noise(fuzzy_checker: FuzzyChecker):
    """
    Tests that the index length we will noise validates to expected noise level
    """

    df = pd.DataFrame(index=list(range(1000)))
    noise_level = 0.62
    chosen_idx = get_index_to_noise(
        data=df,
        noise_level=noise_level,
        randomness_stream=RANDOMNESS0,
        additional_key="test_get_index_to_noise",
        is_column_noise=False,
    )
    # Assert that the proportion of rows to noise is correct
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_get_index_to_noise",
        observed_numerator=len(chosen_idx),
        observed_denominator=len(df),
        target_proportion=noise_level,
    )


def test_vectorized_choice(fuzzy_checker: FuzzyChecker):

    options = ["Supersonics", "Mariners", "Kraken"]
    num_choices = 50_000
    choice_weights = [0.40, 0.35, 0.25]
    picks = vectorized_choice(
        options=options,
        n_to_choose=num_choices,
        randomness_stream=RANDOMNESS0,
        weights=choice_weights,
        additional_key="best_seattle_sports_team",
    )
    picks = pd.Series(picks)

    assert (picks.isin(options)).all()
    # Test weights validate
    for i in range(len(options)):
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_vectorized_choice",
            observed_numerator=(picks == options[i]).sum(),
            observed_denominator=len(picks),
            target_proportion=choice_weights[i],
            name_additional=f"option {options[i]}",
        )


def test_two_d_array_choice(fuzzy_checker: FuzzyChecker):

    best_sports = ["Basketball", "Baseball", "Football"]
    sports = pd.Series(best_sports * 1000)
    options = pd.DataFrame(
        {
            "option1": [
                "Supersonics",
                "Mariners",
                "Ducks",
            ],
            "option2": [
                "Nuggets",
                "Dodgers",
                "Steelers",
            ],
            "option3": [
                "Trail Blazers",
                np.nan,
                "Cougars",
            ],
        },
        index=["Basketball", "Baseball", "Football"],
    )
    choices = two_d_array_choice(
        data=sports,
        options=options,
        randomness_stream=RANDOMNESS0,
        additional_key="2D_best_sports_team",
    )

    assert (choices.isin(options.values.flatten())).all()
    # Assert proportions
    for sport in best_sports:
        num_choices = 3 if sport != "Baseball" else 2
        teams = options.loc[sport, options.loc[sport].notna()].values
        for team in teams:
            fuzzy_checker.fuzzy_assert_proportion(
                name="test_two_d_array_choice",
                observed_numerator=(choices == team).sum(),
                observed_denominator=(sports == sport).sum(),
                target_proportion=1 / num_choices,
                name_additional=f"team {team} for sport {sport}",
            )
