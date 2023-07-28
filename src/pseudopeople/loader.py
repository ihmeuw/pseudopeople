from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.schema_entities import COLUMNS


def load_standard_dataset_file(data_path: Path, user_filters: List[Tuple]) -> pd.DataFrame:
    if data_path.suffix == ".parquet":
        if len(user_filters) == 0:
            # pyarrow.parquet.read_table doesn't accept an empty list
            user_filters = None
        data = pq.read_table(data_path, filters=user_filters).to_pandas()
    else:
        raise DataSourceError(
            f"Source path must be a .parquet file. Provided {data_path.suffix}"
        )
    if not isinstance(data, pd.DataFrame):
        raise DataSourceError(
            f"File located at {data_path} must contain a pandas DataFrame. "
            "Please provide the path to the unmodified root data directory."
        )

    return data


def load_and_prep_1040_data(data_path: dict, user_filters: List[Tuple]) -> pd.DataFrame:
    # Function that loads all tax datasets and formats them to the DATASET.1040 schema

    # Load data
    df_1040 = load_standard_dataset_file(data_path[DatasetNames.TAXES_1040], user_filters)
    breakpoint()
    # We do not want to filter by state for dependents
    for i in range(len(user_filters)):
        if user_filters[i][0] == COLUMNS.mailing_state.name:
            user_filters.pop(i)
    breakpoint()
    df_dependents = load_standard_dataset_file(
        data_path[DatasetNames.TAXES_DEPENDENTS], user_filters
    )

    # Get wide format of dependents - metadata for each guardian's dependents
    dependents_wide = flatten_data(
        data=df_dependents,
        index_cols=[COLUMNS.guardian_id.name, COLUMNS.tax_year.name],
        rank_col=COLUMNS.simulant_id.name,
        value_cols=[
            COLUMNS.simulant_id.name,
            COLUMNS.first_name.name,
            COLUMNS.last_name.name,
            COLUMNS.ssn.name,
            COLUMNS.copy_ssn.name,
        ],
    )
    # Rename tax_dependents columns
    dependents_wide = dependents_wide.add_prefix("dependent_").reset_index()
    # Make sure we have all dependent columns if data does not have a guardian with 4 dependents
    for i in range(2, 5):
        if f"dependent_{i}_first_name" not in dependents_wide.columns:
            for column in [col.name for col in COLUMNS if f"dependent_{i}" in col.name]:
                dependents_wide[column] = np.nan

    # Widen 1040 data (make one row for spouses that are joint filing)
    df_joint_1040 = combine_joint_filers(df_1040)

    # Merge tax dependents onto their guardians - we must do it twice, merge onto each spouse if joint filing
    tax_1040_w_dependents = df_joint_1040.merge(
        dependents_wide,
        how="left",
        left_on=[COLUMNS.simulant_id.name, COLUMNS.tax_year.name],
        right_on=[COLUMNS.guardian_id.name, COLUMNS.tax_year.name],
    )
    # todo: uncomment with mic-4244. Handle columns with dependents for both guardians
    # tax_1040_w_dependents = tax_1040_w_dependents.merge(
    #   dependents_wide, how="left",
    #   left_on=["COLUMNS.spouse_simulant_id.name", "COLUMNS.tax_year.name"],
    #   right_on=["COLUMNS.guardian_id.name", "COLUMNS.tax_year.name"])

    return tax_1040_w_dependents


def flatten_data(
    data: pd.DataFrame,
    index_cols: str,
    rank_col: str,
    value_cols: List[str],
    ascending: bool = False,
) -> pd.DataFrame:
    # Function that takes a dataset and widens (pivots) it to capture multiple metadata columns
    # Example: simulant_id, dependdent_1, dependent_2, dependent_1_name, dependent_2_name, etc...
    data = data.copy()
    # fixme: find a better solution than the following call since applying lambda functions is slow
    data["rank"] = (
        data.groupby(index_cols, group_keys=False)[rank_col]
        .apply(lambda x: x.rank(method="first", ascending=ascending))
        .astype(int)
    )
    # todo: Improve via mic-4244 for random sampling of dependents
    # Choose 4 dependents
    data = data.loc[data["rank"] < 5]
    data["rank"] = data["rank"].astype(str)
    flat = data.pivot(columns="rank", index=index_cols, values=value_cols)
    flat.columns = ["_".join([pair[1], pair[0]]) for pair in flat.columns]

    return flat


def combine_joint_filers(data: pd.DataFrame) -> pd.DataFrame:
    # Get groups
    joint_filers = data.loc[data[COLUMNS.joint_filer.name] == True]
    reference_persons = data.loc[
        data[COLUMNS.relationship_to_reference_person.name] == "Reference person"
    ]
    independent_filers_index = data.index.difference(
        joint_filers.index.union(reference_persons.index)
    )
    # This is a dataframe with all independent filing individuals that are not a reference person
    independent_filers = data.loc[independent_filers_index]

    joint_filers = joint_filers.add_prefix("spouse_")
    # Merge spouses
    reference_persons_wide = reference_persons.merge(
        joint_filers,
        how="left",
        left_on=[COLUMNS.household_id.name, COLUMNS.tax_year.name],
        right_on=[COLUMNS.spouse_household_id.name, COLUMNS.spouse_tax_year.name],
    )
    joint_1040 = pd.concat([reference_persons_wide, independent_filers])

    return joint_1040
