from typing import Any

import numpy as np
import pandas as pd
import yaml
from vivarium import ConfigTree
from vivarium.framework.randomness import RandomnessStream

from pseudopeople.constants import paths
from pseudopeople.utilities import vectorized_choice


def omit_rows(
    form_data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :return:
    """
    # todo actually omit rows
    return form_data


def duplicate_rows(
    form_data: pd.DataFrame,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """

    :param form_data:
    :param configuration:
    :param randomness_stream:
    :return:
    """
    # todo actually duplicate rows
    return form_data


def generate_incorrect_selections(
    column: pd.Series,
    _: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """
    Function that takes a categorical series and applies noise so some values has been replace with other options from
    a list.

    :param column:  A categorical pd.Series
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param additional_key: Key for RandomnessStream
    :returns: pd.Series where data has been noised with other values from a list of possibilities
    """

    selection_type = {
        "employer_state": "state",
        "mailing_address_state": "state",
    }.get(str(column.name), column.name)

    selection_options = pd.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA)

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = selection_options.loc[selection_options[selection_type].notna(), selection_type]
    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{additional_key}_{column.name}_incorrect_select_choice",
    ).to_numpy()

    return pd.Series(new_values, index=column.index)


def generate_within_household_copies(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return column


def swap_months_and_days(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return column


def miswrite_zipcodes(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return column


def miswrite_ages(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """Function to mis-write ages based on perturbation parameters included in
    the config file.

    :param column: pd.Series of ages
    :param configuration: ConfigTree
    :param randomness_stream: Vivarium RandomnessStream
    :param additional_key: additional key used for randomness_stream calls
    :return:
    """
    possible_perturbations = configuration.possible_perturbations
    perturbation_levels = configuration.possible_perturbation_levels
    if not perturbation_levels:
        perturbation_levels = None
    # TODO: Move all these checks upfront
    if perturbation_levels:
        if sum(perturbation_levels) != 1:
            # TODO: Consider adding flexibility here since vectorized_choice will scale.
            raise ValueError(
                "The provided possible_perturbation_levels must sum to 1 but they "
                f"currently sum to {sum(perturbation_levels)}: {perturbation_levels}"
            )
        if len(perturbation_levels) != len(possible_perturbations):
            raise ValueError(
                f"The provided possible perturbation_levels ({perturbation_levels}) must be the "
                f"same length as the provided perturbation probabilities ({possible_perturbations})"
            )
    perturbations = vectorized_choice(
        options=possible_perturbations,
        weights=perturbation_levels,
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{additional_key}_{column.name}_miswrite_ages",
    )
    df = pd.DataFrame({"original_age": column})
    df["age"] = df["original_age"].astype(float).astype(int) + perturbations
    # Reflect negative values to positive
    df.loc[df["age"] < 0, "age"] = -1 * df["age"]
    # If new age == original age, subtract 1
    df.loc[df["age"] == df["original_age"], "age"] -= 1

    return df["age"].astype(str)


def miswrite_numerics(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually duplicate rows
    return column


def generate_nicknames(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate nicknames
    return column


def generate_fake_names(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate fake names
    return column


def generate_phonetic_errors(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate fake names
    return column


def generate_missing_data(column: pd.Series, *_: Any) -> pd.Series:
    """
    Function that takes a column and blanks out all values.

    :param column:  pd.Series of data
    :returns: pd.Series of empty strings with the index of column.
    """

    return pd.Series("", index=column.index)


def generate_typographical_errors(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """Function that takes a column and applies noise to the string values
    representative of keyboard mis-typing.

    :param column:  pd.Series of data
    :param configuration: ConfigTree object containing noising parameters
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN
    :param additional_key: Key for RandomnessStream
    :returns: pd.Series of column with noised data
    """

    with open(paths.QWERTY_ERRORS) as f:
        qwerty_errors = yaml.full_load(f)

    def keyboard_corrupt(truth, corrupted_pr, addl_pr, rng):
        """For each string, loop through each character and determine if
        it is to be corrupted. If so, uniformly choose from the appropriate
        values to mis-type. Also determine which mis-typed characters should
        include the original value and, if it does, include the original value
        after the mis-typed value
        """
        err = ""
        i = 0
        while i < len(truth):
            error_introduced = False
            token = truth[i : (i + 1)]
            if token in qwerty_errors and not error_introduced:
                random_number = rng.uniform()
                if random_number < corrupted_pr:
                    err += rng.choice(qwerty_errors[token])
                    random_number = rng.uniform()
                    if random_number < addl_pr:
                        err += token
                    i += 1
                    error_introduced = True
            if not error_introduced:
                err += truth[i : (i + 1)]
                i += 1
        return err

    token_noise_level = configuration.token_noise_level
    include_original_token_level = configuration.include_original_token_level

    rng = np.random.default_rng(seed=randomness_stream.seed)
    for idx in column.index:
        noised_value = keyboard_corrupt(
            column[idx],
            token_noise_level,
            include_original_token_level,
            rng,
        )
        column[idx] = noised_value

    return column


def generate_ocr_errors(
    column: pd.Series,
    configuration: ConfigTree,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Series:
    """

    :param column:
    :param configuration:
    :param randomness_stream:
    :param additional_key: Key for RandomnessStream
    :return:
    """
    # todo actually generate OCR errors
    return column
