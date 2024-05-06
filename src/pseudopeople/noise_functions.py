from typing import Any

import numpy as np
import pandas as pd
from layered_config_tree import LayeredConfigTree
from vivarium.framework.randomness import RandomnessStream, get_hash

from pseudopeople.configuration import Keys
from pseudopeople.constants import data_values
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.constants.noise_type_metadata import (
    COPY_HOUSEHOLD_MEMBER_COLS,
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
    HOUSING_TYPE_GUARDIAN_DUPLICATION_RELATONSHIP_MAP,
    INT_TO_STRING_COLUMNS,
)
from pseudopeople.data.fake_names import fake_first_names, fake_last_names
from pseudopeople.noise_scaling import (
    load_incorrect_select_options,
    load_nicknames_data,
)
from pseudopeople.utilities import (
    get_index_to_noise,
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
    to_string,
    two_d_array_choice,
    vectorized_choice,
)


def omit_rows(
    dataset_name: str,
    dataset_data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """
    Function that omits rows from a dataset and returns only the remaining rows.  Note that for the ACS and CPS datasets
      we need to account for oversampling in the PRL simulation so a helper function has been hadded here to do so.
    :param dataset_name: Dataset object being noised
    :param dataset_data:  pd.DataFrame of one of the dataset types used in Pseudopeople
    :param configuration: LayeredConfigTree object containing noise level values
    :param randomness_stream: RandomnessStream object to make random selection for noise
    :return: pd.DataFrame with rows from the original dataframe removed
    """

    noise_level = configuration[Keys.ROW_PROBABILITY]
    # Omit rows
    to_noise_index = get_index_to_noise(
        dataset_data,
        noise_level,
        randomness_stream,
        f"{dataset_name}_omit_choice",
    )
    noised_data = dataset_data.loc[dataset_data.index.difference(to_noise_index)]

    return noised_data


def _get_census_omission_noise_levels(
    population: pd.DataFrame,
    base_probability: float = data_values.DO_NOT_RESPOND_BASE_PROBABILITY,
) -> pd.Series:
    """
    Helper function for do_not_respond noising based on demography of age, race/ethnicity, and sex.

    :param population: a dataset containing records of simulants
    :param base_probability: base probability for do_not_respond
    :return: a pd.Series of probabilities
    """
    probabilities = pd.Series(base_probability, index=population.index)
    probabilities += (
        population["race_ethnicity"]
        .astype(str)
        .map(data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_RACE)
    )
    ages = pd.Series(np.arange(population["age"].max() + 1))
    for sex in ["Female", "Male"]:
        effect_by_age_bin = data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE[sex]
        # NOTE: calling pd.cut on a large array with an IntervalIndex is slow,
        # see https://github.com/pandas-dev/pandas/issues/47614
        # Instead, we only pd.cut the unique ages, then do a simpler `.map` on the age column
        age_bins = pd.cut(ages, bins=effect_by_age_bin.index)
        effect_by_age = pd.Series(
            age_bins.map(effect_by_age_bin),
            index=ages,
        )
        sex_mask = population["sex"] == sex
        probabilities[sex_mask] += (
            population[sex_mask]["age"].map(effect_by_age).astype(float)
        )
    probabilities[probabilities < 0.0] = 0.0
    probabilities[probabilities > 1.0] = 1.0
    return probabilities


def apply_do_not_respond(
    dataset_name: str,
    dataset_data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """
    Applies targeted omission based on demographic model for census and surveys.

    :param dataset_name: Dataset object name being noised
    :param dataset_data:  pd.DataFrame of one of the form types used in Pseudopeople
    :param configuration: LayeredConfigTree object containing noise level values
    :param randomness_stream: RandomnessStream object to make random selection for noise
    :return: pd.DataFrame with rows from the original dataframe removed
    """
    required_columns = ("age", "race_ethnicity", "sex")
    missing_columns = [col for col in required_columns if col not in dataset_data.columns]
    if len(missing_columns):
        raise ValueError(
            f"Dataset {dataset_name} is missing required columns: {missing_columns}"
        )

    # do_not_respond noise_levels are based on census
    noise_levels = _get_census_omission_noise_levels(dataset_data)

    # Apply an overall non-response rate of 27.6% for Current Population Survey (CPS)
    if dataset_name == DatasetNames.CPS:
        noise_levels += 0.276

    # Apply user-configured noise level
    configured_noise_level = configuration[Keys.ROW_PROBABILITY]
    default_noise_level = data_values.DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY[dataset_name]
    noise_levels = noise_levels * (configured_noise_level / default_noise_level)

    # Account for ACS and CPS oversampling
    if dataset_name in [DatasetNames.ACS, DatasetNames.CPS]:
        noise_levels = 0.5 + noise_levels / 2

    to_noise_idx = get_index_to_noise(
        dataset_data, noise_levels, randomness_stream, f"do_not_respond_{dataset_name}"
    )
    noised_data = dataset_data.loc[dataset_data.index.difference(to_noise_idx)]

    return noised_data


# def duplicate_rows(
#     dataset_data: pd.DataFrame,
#     configuration: LayeredConfigTree,
#     randomness_stream: RandomnessStream,
# ) -> pd.DataFrame:
#     """

#     :param dataset_data:
#     :param configuration:
#     :param randomness_stream:
#     :return:
#     """
#     # todo actually duplicate rows
#     return dataset_data


def duplicate_with_guardian(
    dataset_name: str,
    dataset_data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    """
    Function that duplicates rows of a dataset. Rows that are duplicated fall into one of three groups of
    dependents that are typically duplicated in administrative data where a dependent lives in a different
    location than their guardian and they show up in the data at both locations. For simplicity, rows will
    be only duplicated once for a maximum of two rows per dependent. When a row is duplicated, one row will
    have the dependent's correct address and the other will have the guardian's address.
    :param dataset_name: Name of the dataset being noised
    :param dataset_data: pd.DataFrame that will be noised
    :param configuration: LayeredConfigTree object containing noise level values. Dict with three key groups for duplication
    :param randomness_stream: RandomnessStream instance to make random selection for noise
    :return: pd.DataFrame with rows from the original dataframe duplicated along with the original
    dataframe itself.
    """

    # Helper function to format group dataframe and merging with their dependentss
    def _merge_dependents_and_guardians(
        dependents_df: pd.DataFrame, full_data: pd.DataFrame
    ) -> pd.DataFrame:
        # Merge dependents with their guardians. We have to merge twice to check
        # if either guardian is living at a separate location from the dependent.
        guardian_1s = full_data.loc[
            full_data["simulant_id"].isin(full_data["guardian_1"]),
            GUARDIAN_DUPLICATION_ADDRESS_COLUMNS + ["simulant_id"],
        ].add_prefix("guardian_1_")
        dependents_and_guardians_df = dependents_df.merge(
            guardian_1s,
            how="left",
            left_on=["guardian_1", "year"],
            right_on=["guardian_1_simulant_id", "guardian_1_year"],
        )
        del guardian_1s
        guardian_2s = full_data.loc[
            full_data["simulant_id"].isin(full_data["guardian_2"]),
            GUARDIAN_DUPLICATION_ADDRESS_COLUMNS + ["simulant_id"],
        ].add_prefix("guardian_2_")
        dependents_and_guardians_df = dependents_and_guardians_df.merge(
            guardian_2s,
            how="left",
            left_on=["guardian_2", "year"],
            right_on=["guardian_2_simulant_id", "guardian_2_year"],
        )
        del guardian_2s

        return dependents_and_guardians_df

    # Get dict of group type and formatted dataframe for that group that should be noised
    formatted_group_data = {}
    # Get dataframe for each dependent group to merge with guardians
    in_households_under_18 = dataset_data.loc[
        (dataset_data["age"] < 18)
        & (dataset_data["housing_type"] == "Household")
        & (dataset_data["guardian_1"].notna())
    ]
    in_college_under_24 = dataset_data.loc[
        (dataset_data["age"] < 24)
        & (dataset_data["housing_type"] == "College")
        & (dataset_data["guardian_1"].notna())
    ]

    # Merge depedents with their guardians
    formatted_group_data[
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18
    ] = _merge_dependents_and_guardians(in_households_under_18, dataset_data)
    formatted_group_data[
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24
    ] = _merge_dependents_and_guardians(in_college_under_24, dataset_data)
    # Note: We have two dicts (configuration and formatted_group_data) at this point that have
    # the key for the group and then a dataframe for that group or the group and the configured
    # noise level

    noised_data = []
    for group, group_df in formatted_group_data.items():
        # Get index groups that can be noised based on dependent and guardian(s) addresses
        both_different_index = group_df.index[
            (group_df["household_id"] != group_df["guardian_1_household_id"])
            & (group_df["guardian_1_household_id"].notna())
            & (group_df["household_id"] != group_df["guardian_2_household_id"])
            & (group_df["guardian_2_household_id"].notna())
        ]
        # Choose which guardian to copy when dependent lives in different address from both guardains
        choices = randomness_stream.choice(
            both_different_index,
            choices=["guardian_1", "guardian_2"],
            additional_key=f"{dataset_name}_duplicate_with_guardian_{group}_guardian_choice",
        )
        group_df.loc[both_different_index, "copy_guardian"] = choices
        # Get remaining dependents that live in different address from one of their guardians
        guardian_1_different_index = group_df.index[
            (group_df["household_id"] != group_df["guardian_1_household_id"])
            & (group_df["guardian_1_household_id"].notna())
        ].difference(choices.index)
        group_df.loc[guardian_1_different_index, "copy_guardian"] = "guardian_1"
        guardian_2_different_index = group_df.index[
            (group_df["household_id"] != group_df["guardian_2_household_id"])
            & (group_df["guardian_2_household_id"].notna())
        ].difference(choices.index)
        group_df.loc[guardian_2_different_index, "copy_guardian"] = "guardian_2"

        # Noise data
        to_noise_index = get_index_to_noise(
            group_df,
            configuration[group],
            randomness_stream,
            f"{dataset_name}_duplicate_with_guardian_{group}",
        )
        # Copy over address information from guardian to dependent
        for guardian in ["guardian_1", "guardian_2"]:
            index_to_copy = to_noise_index.intersection(
                group_df.index[group_df["copy_guardian"] == guardian]
            )
            # Don't try to copy if there are no rows to copy
            if index_to_copy.empty:
                continue
            noised_group_df = group_df.loc[index_to_copy]
            noised_group_df[GUARDIAN_DUPLICATION_ADDRESS_COLUMNS] = group_df.loc[
                index_to_copy,
                [f"{guardian}_" + column for column in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS],
            ]
            noised_data.append(noised_group_df)

    # Combine all noised dataframes
    if not noised_data:
        return dataset_data
    else:
        noised_data = pd.concat(noised_data)
        # Update relationship to reference person for duplicated simulants based on housing type
        noised_data["relationship_to_reference_person"] = noised_data["housing_type"].map(
            HOUSING_TYPE_GUARDIAN_DUPLICATION_RELATONSHIP_MAP
        )
        # Clean columns
        noised_data = noised_data[dataset_data.columns]
        # Fix to help keep optimization code for noise.py
        index_start_value = dataset_data.index.max() + 1
        noised_data.index = range(index_start_value, index_start_value + len(noised_data))
        dataset_data = pd.concat([dataset_data, noised_data])

    return dataset_data


def choose_wrong_options(
    data: pd.DataFrame,
    _: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that takes a categorical series and applies noise so some values has been replace with other options from
    a list.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param _: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :returns: pd.Series where data has been noised with other values from a list of possibilities
    """

    selection_type = {
        "employer_state": "state",
        "mailing_address_state": "state",
    }.get(str(column_name), column_name)

    selection_options = load_incorrect_select_options()

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = selection_options.loc[selection_options[selection_type].notna(), selection_type]
    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(data),
        randomness_stream=randomness_stream,
        additional_key=f"{column_name}_incorrect_select_choice",
    ).to_numpy()

    return pd.Series(new_values, index=data.index, name=column_name)


def copy_from_household_member(
    data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param _: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :returns: pd.Series where data has been noised with other values from a list of possibilities
    """

    copy_values = data[COPY_HOUSEHOLD_MEMBER_COLS[column_name]]
    column = pd.Series(copy_values, index=data.index, name=column_name)
    return column


def swap_months_and_days(
    data: pd.DataFrame,
    _: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that swaps month and day of dates.

    :param data: A pandas dataframe containing necessary columns for column noise
    :param _: LayeredConfigTree object containing noise level values
    :param randomness_stream: Randomness Stream object for random choices using vivarium CRN framework
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: Noised pd.Series where some dates have month and day swapped.
    """
    from pseudopeople.schema_entities import DATASETS, DATEFORMATS

    date_format = DATASETS.get_dataset(dataset_name).date_format

    column = data[column_name]
    if date_format == DATEFORMATS.YYYYMMDD:  # YYYYMMDD
        year = column.str[:4]
        month = column.str[4:6]
        day = column.str[6:]
        noised = year + day + month
    elif date_format == DATEFORMATS.MM_DD_YYYY:  # MM/DD/YYYY
        year = column.str[6:]
        month = column.str[:3]
        day = column.str[3:6]
        noised = day + month + year
    elif date_format == DATEFORMATS.MMDDYYYY:  # MMDDYYYY
        year = column.str[4:]
        month = column.str[:2]
        day = column.str[2:4]
        noised = day + month + year
    else:
        raise ValueError(f"Invalid date format in {dataset_name}.")

    return noised


def write_wrong_zipcode_digits(
    data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that noises a 5 digit zipcode

    :param data: A pandas dataframe containing necessary columns for column noise
    :param configuration:  Config tree object at column node.
    :param randomness_stream:  RandomnessStream object from Vivarium framework
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series of noised zipcodes
    """

    column = data[column_name]
    str_len = column.str.len()
    if (str_len != 5).sum() > 0:
        # TODO: This is a BAD error message. It should never appear and if it
        #   does, the user shouldn't be checking the simulated population data.
        raise ValueError(
            "Zipcode data contains zipcodes that are not 5 digits long. Please check simulated population data."
        )

    rng = np.random.default_rng(
        get_hash(f"{randomness_stream.seed}_{column_name}_write_wrong_zipcode_digits")
    )
    shape = (len(column), 5)

    # todo: Update when vectorized choice is improved
    possible_replacements = np.array(list("0123456789"))
    # Scale up noise levels to adjust for inclusive sampling with all numbers
    scaleup_factor = 1 / (1 - (1 / len(possible_replacements)))
    # Get configuration values for each piece of 5 digit zipcode
    digit_probabilities = scaleup_factor * np.array(
        configuration[Keys.ZIPCODE_DIGIT_PROBABILITIES]
    )
    replace = rng.random(shape) < digit_probabilities
    num_to_replace = replace.sum()
    random_digits = rng.choice(possible_replacements, num_to_replace)

    # https://stackoverflow.com/a/9493192/
    # Changing this to a U5 numpy string type means that each string will have exactly 5 characters.
    # view("U1") then reinterprets this memory as an array of individual (Unicode) characters.
    same_len_col_exploded = column.values.astype("U5").view("U1").reshape(shape)
    same_len_col_exploded[replace] = random_digits
    return pd.Series(
        same_len_col_exploded.view("U5").reshape(len(column)),
        index=column.index,
        name=column.name,
    )


def misreport_ages(
    data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """Function to mis-write ages based on perturbation parameters included in
    the config file.

    :param data: A pandas dataframe containing necessary columns for column noise
    :param configuration:  Config tree object at column node.
    :param randomness_stream:  RandomnessStream object from Vivarium framework
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series with some values noised from the original
    """

    column = data[column_name]
    possible_perturbations = configuration[Keys.POSSIBLE_AGE_DIFFERENCES].to_dict()
    perturbations = vectorized_choice(
        options=list(possible_perturbations.keys()),
        weights=list(possible_perturbations.values()),
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{column_name}_{column.name}_miswrite_ages",
    )
    new_values = column.astype(int) + perturbations
    # Reflect negative values to positive
    new_values[new_values < 0] *= -1
    # If new age == original age, subtract 1
    new_values[new_values == column.astype(int)] -= 1

    return new_values


def write_wrong_digits(
    data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that noises numeric characters in a series.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream

    returns: pd.Series with some numeric values experiencing noise.
    """
    column = data[column_name]
    if column.empty:
        return column
    # This is a fix to not replacing the original token for noise options
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY] / 0.9
    rng = np.random.default_rng(
        get_hash(f"{randomness_stream.seed}_{column_name}_write_wrong_digits")
    )
    column = to_string(column)

    max_str_length = column.str.len().max()

    possible_replacements = np.array(list("0123456789"))

    # https://stackoverflow.com/a/9493192/
    # Changing this to a numpy (not Python) string type means that it will have a fixed
    # number of characters, equal to the longest string in the array.
    # view("U1") then reinterprets this memory as an array of individual (Unicode) characters.
    same_len_col_exploded = (
        column.values.astype(str).view("U1").reshape((len(column), max_str_length))
    )
    # Surprisingly, Numpy does not provide a computationally efficient way to do
    # this check for which characters are eligible.
    # A head-to-head comparison found np.isin to be orders of magnitude slower than using Pandas here.
    # Also, np.isin fails silently with sets (see https://numpy.org/doc/stable/reference/generated/numpy.isin.html)
    # so be careful if testing that in the future!
    is_number = (
        pd.DataFrame(same_len_col_exploded).isin(set(possible_replacements)).to_numpy()
    )

    replace = np.zeros_like(is_number, dtype=bool)
    replace[is_number] = rng.random(is_number.sum()) < token_noise_level
    num_to_replace = replace.sum()
    random_digits = rng.choice(possible_replacements, num_to_replace)

    same_len_col_exploded[replace] = random_digits
    noised_column = same_len_col_exploded.view(f"U{max_str_length}").reshape(len(column))

    return pd.Series(noised_column, index=column.index, name=column.name)


def use_nicknames(
    data: pd.DataFrame,
    _: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that replaces a name with a choice of potential nicknames.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param _: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series of nicknames replacing original names
    """
    nicknames = load_nicknames_data()
    nickname_eligible_names = set(nicknames.index)
    column = data[column_name]
    have_nickname_idx = column.index[column.isin(nickname_eligible_names)]
    noised = two_d_array_choice(
        column.loc[have_nickname_idx],
        nicknames,
        randomness_stream,
        f"{column_name}_use_nicknames",
    )
    column.loc[have_nickname_idx] = noised
    return column


def use_fake_names(
    data: pd.DataFrame,
    _: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param _: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return:
    """
    column = data[column_name]
    fake_first = fake_first_names
    fake_last = fake_last_names
    fake_names = {
        "first_name": fake_first,
        "middle_name": fake_first,
        "last_name": fake_last,
        "spouse_first_name": fake_first,
        "spouse_last_name": fake_last,
        "dependent_1_first_name": fake_first,
        "dependent_1_last_name": fake_last,
        "dependent_2_first_name": fake_first,
        "dependent_2_last_name": fake_last,
        "dependent_3_first_name": fake_first,
        "dependent_3_last_name": fake_last,
        "dependent_4_first_name": fake_first,
        "dependent_4_last_name": fake_last,
    }
    options = fake_names[column_name]

    new_values = vectorized_choice(
        options=options,
        n_to_choose=len(column),
        randomness_stream=randomness_stream,
        additional_key=f"{column_name}_fake_names",
    )
    return pd.Series(new_values, index=column.index, name=column.name)


def make_phonetic_errors(
    data: pd.Series,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: Any,
) -> pd.Series:
    """

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series of noised data
    """

    # Load phonetic errors
    phonetic_errors = load_phonetic_errors()

    return _corrupt_tokens(
        phonetic_errors,
        data[column_name].astype(str),
        configuration[Keys.TOKEN_PROBABILITY],
        randomness_stream,
        addl_key=f"{column_name}_make_phonetic_errors",
    )


def leave_blanks(
    data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    Function that takes a column and blanks out all values.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    """
    return pd.Series(np.nan, index=data.index, name=column_name)


def make_typos(
    data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """Function that applies noise to the string values
    representative of keyboard mistyping.

    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :returns: pd.Series of column with noised data
    """

    qwerty_errors = load_qwerty_errors_data()
    qwerty_errors_eligible_chars = set(qwerty_errors.index)

    column = data[column_name]
    if column.empty:
        return column
    column = column.astype(str)
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY]
    # TODO: remove this hard-coding
    include_token_probability_level = 0.1
    rng = np.random.default_rng(
        seed=get_hash(f"{randomness_stream.seed}_{column_name}_make_typos")
    )

    same_len_col_exploded = (
        # Somewhat counterintuitively, .astype(str) turns the column into a numpy,
        # fixed-length string type, U#, where # is the length of the longest string.
        column.values.astype(str)
        # Split into individual characters
        .view("U1").reshape((len(column), -1))
        # https://stackoverflow.com/a/9493192/
        # NOTE: Surprisingly, casting this to "object" (Python string type) seems to make
        # the rest of this method faster. Without this cast, you have to use np.char.add to concatenate,
        # which seems very slow for some reason?
        .astype("object")
    )

    # Surprisingly, Numpy does not provide a computationally efficient way to do
    # this check for which characters are eligible.
    # A head-to-head comparison found np.isin to be orders of magnitude slower than using Pandas here.
    # Also, np.isin fails silently with sets (see https://numpy.org/doc/stable/reference/generated/numpy.isin.html)
    # so be careful if testing that in the future!
    is_typo_option = (
        pd.DataFrame(same_len_col_exploded).isin(qwerty_errors_eligible_chars).to_numpy()
    )
    replace = np.zeros_like(is_typo_option)
    replace[is_typo_option] = rng.random(is_typo_option.sum()) < token_noise_level
    keep_original = np.zeros_like(replace)
    keep_original[replace] = rng.random(replace.sum()) < include_token_probability_level

    # Apply noising
    to_replace = same_len_col_exploded[replace]
    replace_random = rng.random(replace.sum())
    number_of_options = qwerty_errors.count(axis=1)
    replace_option_index = np.floor(
        replace_random * number_of_options.loc[to_replace].to_numpy()
    )
    originals_to_keep = same_len_col_exploded[keep_original]
    # Numpy does not have an efficient way to do this merge operation
    same_len_col_exploded[replace] = (
        pd.DataFrame({"to_replace": to_replace, "replace_option_index": replace_option_index})
        .reset_index()
        .merge(
            qwerty_errors.stack().rename("replacement"),
            left_on=["to_replace", "replace_option_index"],
            right_index=True,
        )
        # Merge does not return the results in order
        .sort_values("index")
        .replacement.to_numpy()
    )

    same_len_col_exploded[keep_original] += originals_to_keep

    noised_column = np.sum(same_len_col_exploded, axis=1)

    return pd.Series(noised_column, index=column.index, name=column.name)


def make_ocr_errors(
    data: pd.DataFrame,
    configuration: LayeredConfigTree,
    randomness_stream: RandomnessStream,
    dataset_name: str,
    column_name: str,
) -> pd.Series:
    """
    :param data:  A pandas dataframe containing necessary columns for column noise
    :param configuration: LayeredConfigTree with rate at which to blank the data in column.
    :param randomness_stream:  RandomnessStream to utilize Vivarium CRN.
    :param column_name: String for column that will be noised, will be the key for RandomnessStream
    :return: pd.Series of noised data
    """

    # Load OCR error dict
    ocr_errors = load_ocr_errors()

    return _corrupt_tokens(
        ocr_errors,
        data[column_name].astype(str),
        configuration[Keys.TOKEN_PROBABILITY],
        randomness_stream,
        addl_key=f"{column_name}_make_ocr_errors",
    )


def _corrupt_tokens(
    errors: pd.DataFrame,
    column: pd.Series,
    token_probability: float,
    randomness_stream: RandomnessStream,
    addl_key: str,
) -> pd.Series:
    """
    Performs token-level corruption on a string Series when the tokens to corrupt
    (and the tokens they get corrupted to) can be more than one character long.
    A token does not need to be the same length as a token it gets corrupted to.
    When both kinds of tokens are always one character, things are simpler,
    and a faster algorithm can be used; see typographic noise for an example of this.

    `errors` is a pandas DataFrame where the index contains tokens that are eligible to
    be corrupted, and the columns contain the corruptions that are possible.
    The column names do not matter.
    """
    # NOTE: This first section depends only on `errors` and could be cached -- we might
    # consider moving it into the load_* functions.
    # However, the amount of work here scales only with the size of `errors` and not
    # with the data.
    # It isn't possible to just make this a cached helper since `errors` is a DataFrame,
    # which is unhashable.
    max_token_length = errors.index.str.len().max()
    errors_eligible_tokens = (
        pd.Series(errors.index).groupby(errors.index.str.len()).apply(set).to_dict()
    )
    number_of_options = errors.count(axis=1)
    # Convert to 1D array for easy indexing
    errors_array = errors.values.reshape(-1)
    # Keep track of where in that 1D array the corruption options for each corruptable
    # token start
    errors_array_index_by_string = pd.Series(
        len(errors.columns) * np.array(range(len(errors))), index=errors.index
    )

    lengths = column.str.len().values

    same_len_col_exploded = (
        column
        # Convert to numpy string dtype
        .values.astype(str)
        .view("U1")
        .reshape((len(column), lengths.max()))
    )

    rng = np.random.default_rng(seed=get_hash(f"{randomness_stream.seed}_{addl_key}"))

    # NOTE: Somewhat surprisingly, this seemed to perform better using Python string types than NumPy types.
    # Perhaps worth more investigation in the future.
    # Note that each item in result can be up to the number of characters of the *longest* corrupted token.
    result = np.empty(same_len_col_exploded.shape, dtype=str).astype("object")
    # Tracks the next character index where we have to pay attention to a given string.
    # When we corrupt a token, we skip to the end of that corrupted token before corrupting anything new,
    # so we don't have to consider that string again until we get to the end of it.
    next_due = np.zeros(len(column))
    # We proceed by character through all the strings at once
    for i in range(same_len_col_exploded.shape[1]):
        error_introduced = np.zeros(len(column), dtype=bool)
        assert np.all(next_due >= i)
        due = next_due == i

        # Longer tokens to-be-corrupted take precedence over shorter ones
        for token_length in range(max_token_length, 0, -1):
            if i + token_length > same_len_col_exploded.shape[1]:
                continue

            can_be_corrupted = ~error_introduced & due

            # Is the string already over at this index?
            # If so, we can ignore it.
            # NOTE: From here on, all the boolean arrays are implicitly
            # cumulative; that is, "long_enough" really means that it
            # is due, not already corrupted, *and* long enough. This allows us to short-circuit,
            # e.g. only check the length of strings that are due, and so on.
            long_enough = np.zeros(len(column), dtype=bool)
            long_enough[can_be_corrupted] = i + token_length <= lengths[can_be_corrupted]
            if long_enough.sum() == 0:
                continue
            del can_be_corrupted

            # Collect the tokens of the given length that start at this index.
            tokens = np.empty(len(column), dtype=f"U{token_length}")
            tokens[long_enough] = (
                same_len_col_exploded[long_enough, i : (i + token_length)]
                .view(f"U{token_length}")
                .reshape(long_enough.sum())
            )

            # Are these tokens that have corruptions?
            eligible = np.zeros(len(column), dtype=bool)
            eligible[long_enough] = (
                pd.Series(tokens[long_enough])
                .isin(errors_eligible_tokens.get(token_length, set()))
                .values
            )
            if eligible.sum() == 0:
                continue
            del long_enough

            # If so, should we corrupt them?
            corrupted = np.zeros(len(column), dtype=bool)
            corrupted[eligible] = rng.random(eligible.sum()) < token_probability
            if corrupted.sum() == 0:
                continue
            del eligible

            # If so, which corruption should we choose?
            # There is only a meaningful decision to make here when there are multiple
            # options for a given token, so we avoid drawing randomness when there is
            # only one.
            to_corrupt = tokens[corrupted]
            corrupted_token_index = np.zeros(len(to_corrupt), dtype=int)
            num_options = number_of_options.loc[to_corrupt].to_numpy()
            multiple_options = num_options > 1
            corrupted_token_index[multiple_options] = np.floor(
                rng.random(multiple_options.sum()) * num_options[multiple_options]
            )
            # Get the actual string corresponding to the corrupted token chosen.
            # First, find the index in the errors array corresponding to the first corruption
            # of the token to be corrupted.
            # Then, add the offset of the corrupted token index.
            errors_array_indices = (
                errors_array_index_by_string.loc[to_corrupt].to_numpy()
                + corrupted_token_index
            )
            result[corrupted, i] = errors_array[errors_array_indices]
            # Will not be due again until we reach the end of this token
            next_due[corrupted] = i + token_length
            error_introduced[corrupted] = True

        # Strings that are not "due" -- that is, we already corrupted a source
        # token that included the current character index -- do not even need
        # the original character to be added to the result (it was *replaced*
        # by the corrupted token).
        use_original_char = ~error_introduced & due
        result[use_original_char, i] = same_len_col_exploded[use_original_char, i]
        next_due[use_original_char] = i + 1

    # "Un-explode" (re-concatenate) each string from its pieces.
    return pd.Series(result.sum(axis=1), index=column.index)
