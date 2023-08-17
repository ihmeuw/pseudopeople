from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pseudopeople.constants.metadata import COPY_HOUSEHOLD_MEMBER_COLS, DatasetNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.schema_entities import COLUMNS
from pseudopeople.utilities import engine_implementations

if TYPE_CHECKING:
    from pseudopeople.utilities import DATAFRAME, ENGINE


def load_standard_dataset_file(
    data_path: Path, user_filters: List[Tuple], engine: ENGINE = "pandas"
) -> DATAFRAME:
    if engine == "pandas":
        if data_path.suffix != ".parquet":
            raise DataSourceError(
                f"Source path must be a .parquet file. Provided {data_path.suffix}"
            )

        if len(user_filters) == 0:
            # pyarrow.parquet.read_table doesn't accept an empty list
            user_filters = None
        data = pq.read_table(data_path, filters=user_filters).to_pandas()

        if not isinstance(data, pd.DataFrame):
            raise DataSourceError(
                f"File located at {data_path} must contain a pandas DataFrame. "
                "Please provide the path to the unmodified root data directory."
            )

        return data
    else:
        # Modin
        import modin.pandas as mpd

        # NOTE: Modin doesn't work with PosixPath types
        # TODO: released versions of Modin can't actually distribute `filters`, see https://github.com/modin-project/modin/issues/5509
        # So for now, modin doesn't actually get us distributed loading of the data, and it all needs to fit into
        # memory on a single machine, which mostly beats the point.
        # This has been fixed in the master branch of Modin's GitHub, but we can't use a bleeding edge version
        # because it requires pandas>=2.0.0 which Vivarium doesn't support yet.
        # For now, install modin from the modin_22_backport_parquet_filters branch at https://github.com/zmbc/modin
        return mpd.read_parquet(str(data_path), filters=user_filters)


def load_and_prep_1040_data(
    data_path: dict, user_filters: List[Tuple], engine: ENGINE = "pandas"
) -> DATAFRAME:
    # Function that loads all tax datasets and formats them to the DATASET.1040 schema

    # Load data
    df_1040 = load_standard_dataset_file(
        data_path[DatasetNames.TAXES_1040], user_filters, engine=engine
    )
    df_dependents = load_standard_dataset_file(
        data_path[DatasetNames.TAXES_DEPENDENTS],
        user_filters,
        engine=engine,
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
            COLUMNS.copy_ssn.name,
        ],
    )
    # Rename tax_dependents columns
    dependents_wide = dependents_wide.add_prefix("dependent_").reset_index()
    # Widen 1040 data (make one row for spouses that are joint filing)
    df_joint_1040 = combine_joint_filers(df_1040, engine=engine)

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
    data: DATAFRAME,
    index_col: str,
    rank_col: str,
    value_cols: List[str],
    ascending: bool = False,
) -> DATAFRAME:
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


def combine_joint_filers(data: DATAFRAME, engine: ENGINE = "pandas") -> DATAFRAME:
    # Get groups
    joint_filers = data.loc[data["joint_filer"] == True]
    reference_persons = data.loc[
        data[COLUMNS.relation_to_reference_person.name] == "Reference person"
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
        left_on=COLUMNS.household_id.name,
        right_on=COLUMNS.spouse_household_id.name,
    )

    _, pandas_like = engine_implementations(engine)

    joint_1040 = pandas_like.concat(
        [reference_persons_wide, independent_filers]
    ).reset_index()

    return joint_1040
