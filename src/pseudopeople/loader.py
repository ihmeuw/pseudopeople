from pathlib import Path
from typing import List, Tuple

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
    df_dependents = load_standard_dataset_file(
        data_path[DatasetNames.TAXES_DEPENDENTS], user_filters
    )

    # Get wide format of dependents - metadata for each guardian's dependents
    dependents_wide = flatten_data(
        data=df_dependents,
        index_col=COLUMNS.guardian_id.name,
        rank_col=COLUMNS.simulant_id.name,
        value_cols=[
            COLUMNS.simulant_id.name,
            COLUMNS.first_name.name,
            COLUMNS.last_name.name,
            COLUMNS.ssn.name,
        ],
    )
    # Rename tax_dependents columns
    dependents_wide = dependents_wide.add_prefix("dependent_").reset_index()
    # Widen 1040 data (make one row for spouses that are joint filing)
    df_joint_1040 = combine_joint_filers(df_1040)

    # Merge tax dependents onto their guardians - we must do it twice, merge onto each spouse if joint filing
    tax_1040_w_dependents = df_joint_1040.merge(
        dependents_wide,
        how="left",
        left_on=COLUMNS.simulant_id.name,
        right_on=COLUMNS.guardian_id.name,
    )
    # todo: uncomment with mic-4244. Handle columns with dependents for both guardians
    # tax_1040_w_dependents = tax_1040_w_dependents.merge(dependents_wide, how="left", left_on="COLUMNS.spouse_simulant_id.name", right_on="COLUMNS.guardian_id.name")

    return tax_1040_w_dependents


def flatten_data(
    data: pd.DataFrame,
    index_col: str,
    rank_col: str,
<<<<<<< HEAD
<<<<<<< HEAD
    value_cols: List[str],
=======
    value_cols: list[str],
>>>>>>> e70d81a (Format 1040 data, adds formatting functions to new loader module)
=======
    value_cols: List[str],
>>>>>>> d4ed2f4 (Typing)
    ascending: bool = False,
) -> pd.DataFrame:
    # Function that takes a dataset and widens (pivots) it to capture multiple metadata columns
    # Example: simulant_id, dependdent_1, dependent_2, dependent_1_name, dependent_2_name, etc...
    data = data.copy()
    data["rank"] = (
        data.groupby(index_col, group_keys=False)[rank_col]
        .apply(lambda x: x.rank(method="first", ascending=ascending))
        .astype(int)
    )
    # todo: Improve via mic-4244 for random sampling of dependents
    # Choose 4 dependents
    data = data.loc[data["rank"] < 5]
    data["rank"] = data["rank"].astype(str)
    flat = data.pivot(columns="rank", index=index_col, values=value_cols)
    flat.columns = ["_".join([pair[1], pair[0]]) for pair in flat.columns]
    return flat


def combine_joint_filers(data: pd.DataFrame) -> pd.DataFrame:
    # Get groups
    joint_filers = data.loc[data["joint_filer"] == True]
    reference_persons = data.loc[
        data[COLUMNS.relation_to_reference_person.name] == "Reference person"
    ]
<<<<<<< HEAD
    independent_filers_index = data.index.difference(
        joint_filers.index.union(reference_persons.index)
    )
    # This is a dataframe with all independent filing individuals that are not a reference person
    independent_filers = data.loc[independent_filers_index]

    joint_filers = joint_filers.add_prefix("spouse_")
    # Merge spouses
    reference_persons_wide = reference_persons.merge(
=======
    no_spouses_index = data.index.difference(
        joint_filers.index.union(reference_persons.index)
    )
    no_spouses = data.loc[no_spouses_index]

    joint_filers = joint_filers.add_prefix("spouse_")
    # Merge spouses
    spouses = reference_persons.merge(
>>>>>>> e70d81a (Format 1040 data, adds formatting functions to new loader module)
        joint_filers,
        left_on=COLUMNS.household_id.name,
        right_on=COLUMNS.spouse_household_id.name,
    )
<<<<<<< HEAD
    joint_1040 = pd.concat([reference_persons_wide, independent_filers]).reset_index()
=======
    joint_1040 = pd.concat([spouses, no_spouses]).reset_index()
>>>>>>> e70d81a (Format 1040 data, adds formatting functions to new loader module)

    return joint_1040
