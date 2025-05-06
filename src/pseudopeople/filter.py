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


def get_data_filters(
    dataset_schema: DatasetSchema, year: int | None = 2020, state: str | None = None
) -> Sequence[DataFilter]:
    filters = []
    if dataset_schema.has_state_filter and state is not None:
        state_column = cast(str, dataset_schema.state_column_name)
        filters.append(DataFilter(state_column, "==", get_state_abbreviation(state)))

    if year is not None:
        try:
            if dataset_schema.has_year_lower_filter:
                date_lower_filter = DataFilter(
                    dataset_schema.date_column_name,
                    ">=",
                    pd.Timestamp(year=year, month=1, day=1),
                )
                filters.append(date_lower_filter)

            if dataset_schema.has_year_upper_filter:
                date_upper_filter = DataFilter(
                    dataset_schema.date_column_name,
                    "<=",
                    pd.Timestamp(year=year, month=12, day=31),
                )
                filters.append(date_upper_filter)
        except (pd.errors.OutOfBoundsDatetime, ValueError):
            raise ValueError(f"Invalid year provided: '{year}'")

        if dataset_schema.has_exact_year_filter:
            filters.append(DataFilter(dataset_schema.date_column_name, "==", year))

    return filters
