import numpy as np
import pandas as pd

from pseudopeople.constants import metadata, paths


def scale_choose_wrong_option(data: pd.DataFrame, column_name: str) -> float:
    """
    Function to scale noising for choose_wrong_option to adjust for the possibility
    of noising with the original values.
    """

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = get_options_for_column(column_name)

    # Scale to adjust for possibility of noising with original value
    noise_scaling_value = 1 / (1 - 1 / len(options))

    return noise_scaling_value


def scale_nicknames(data: pd.DataFrame, column_name: str) -> float:
    # Constant calculated by number of names with nicknames / number of names used in PRL name mapping
    nicknames = load_nicknames_data()
    proportion_have_nickname = (
        data[column_name].isin(nicknames.index).sum() / data[column_name].notna().sum()
    )
    if proportion_have_nickname == 0.0:
        return 0.0
    return 1 / proportion_have_nickname


def scale_copy_from_household_member(data: pd.DataFrame, column_name: str) -> float:
    copy_column = data[metadata.COPY_HOUSEHOLD_MEMBER_COLS[column_name]]
    eligible_idx = copy_column.index[(copy_column != "") & (copy_column.notna())]
    proportion_eligible = len(eligible_idx) / len(data)
    if proportion_eligible == 0.0:
        return 0.0
    return 1 / proportion_eligible


####################
# Helper functions #
####################


def load_nicknames_data():
    # Load and format nicknames dataset
    nicknames = pd.read_csv(paths.NICKNAMES_DATA)
    nicknames = nicknames.apply(lambda x: x.astype(str).str.title()).set_index("name")
    nicknames = nicknames.replace("Nan", np.nan)
    return nicknames


def get_options_for_column(column_name: str) -> pd.Series:
    """
    For a column that has a set list of options, returns that set of options as
    a Series.
    Should only be passed a column that has options (i.e. should not be a free-form
    string column such as first name, or a numeric column), or it will error.
    """
    from pseudopeople.schema_entities import COLUMNS

    selection_type = {
        COLUMNS.employer_state.name: COLUMNS.state.name,
        COLUMNS.mailing_state.name: COLUMNS.state.name,
    }.get(column_name, column_name)

    selection_options = pd.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA)
    return selection_options.loc[selection_options[selection_type].notna(), selection_type]
