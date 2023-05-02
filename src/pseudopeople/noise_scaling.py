import numpy as np
import pandas as pd

from pseudopeople.constants import paths, metadata


def noise_scaling_incorrect_selection(column: pd.Series) -> float:
    """
    Function to scale noising for incorrect selection to adjust for the possibility of noising with the original values.
    """
    from pseudopeople.schema_entities import COLUMNS
    selection_type = {
        COLUMNS.employer_state.name: COLUMNS.state.name,
        COLUMNS.mailing_address_state.name: COLUMNS.state.name,
    }.get(column.name, column.name)

    selection_options = pd.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA)
    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = selection_options.loc[selection_options[selection_type].notna(), selection_type]

    # Scale to adjust for possibility of noising with original value
    noise_scaling_value = 1 / (1 - 1 / len(options))

    return noise_scaling_value


def scale_nicknames(column: pd.Series) -> float:
    # Constant calculated by number of names with nicknames / number of names used in PRL name mapping
    nicknames = _load_nicknames_data()
    proportion_have_nickname = column.isin(nicknames.index).mean()
    return 1 / proportion_have_nickname


####################
# Helper functions #
####################

def _load_nicknames_data():
    # Load and format nicknames dataset
    nicknames = pd.read_csv(paths.NICKNAMES_DATA)
    nicknames = nicknames.apply(lambda x: x.astype(str).str.title()).set_index("name")
    nicknames = nicknames.replace("Nan", np.nan)
    return nicknames
