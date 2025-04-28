from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pandas as pd

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.schema_entities import DatasetSchema
from pseudopeople.utilities import get_state_abbreviation


@dataclass
class DataFilter:
    column_name: str
    operator: str
    value: str | int | pd.Timestamp

    def to_tuple(self) -> tuple[str, str, str | int | pd.Timestamp]:
        return self.column_name, self.operator, self.value


def get_generate_data_filters(
    dataset_schema: DatasetSchema, year: int | None = 2020, state: str | None = None
) -> Sequence[DataFilter]:
    filters = []

    # add year filter for SSA
    if dataset_schema.name == DatasetNames.SSA:
        if year is not None:
            try:
                filters.append(
                    DataFilter(
                        dataset_schema.date_column_name,
                        "<=",
                        pd.Timestamp(year=year, month=12, day=31),
                    )
                )
            except (pd.errors.OutOfBoundsDatetime, ValueError):
                raise ValueError(f"Invalid year provided: '{year}'")
    # add state filters except for SSA which does not have a state column
    else:
        if state is not None:
            state_column = cast(str, dataset_schema.state_column_name)
            filters.append(DataFilter(state_column, "==", get_state_abbreviation(state)))

    # add non-SSA year filters
    if dataset_schema.name == DatasetNames.ACS or dataset_schema.name == DatasetNames.CPS:
        if year is not None:
            try:
                date_lower_filter = DataFilter(
                    dataset_schema.date_column_name,
                    ">=",
                    pd.Timestamp(year=year, month=1, day=1),
                )
                date_upper_filter = DataFilter(
                    dataset_schema.date_column_name,
                    "<=",
                    pd.Timestamp(year=year, month=12, day=31),
                )
                filters.extend([date_lower_filter, date_upper_filter])
            except (pd.errors.OutOfBoundsDatetime, ValueError):
                raise ValueError(f"Invalid year provided: '{year}'")
    else:
        if year is not None and dataset_schema.name != DatasetNames.SSA:
            filters.append(DataFilter(dataset_schema.date_column_name, "==", year))

    return filters
