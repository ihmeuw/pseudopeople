from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError


def load_data(
    input_data: Union[Path, pd.DataFrame], user_filters: List[Tuple]
) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        data = filter_data(input_data, user_filters)
    else:
        data = load_standard_dataset_file(input_data, user_filters)

    return data


def filter_data(input_data: pd.DataFrame, user_filters: List[Tuple]) -> pd.DataFrame:
    if len(user_filters) > 0:
        for filter in user_filters:
            input_data = input_data.query(f"{filter[0]} {filter[1]} '{filter[2]}'")

    return input_data


def load_standard_dataset_file(
    input_data: Union[Path, pd.DataFrame], user_filters: List[Tuple]
) -> pd.DataFrame:
    if input_data.suffix == ".parquet":
        if len(user_filters) == 0:
            # pyarrow.parquet.read_table doesn't accept an empty list
            user_filters = None
        data = pq.read_table(input_data, filters=user_filters).to_pandas()
    else:
        raise DataSourceError(
            f"Source path must be a .parquet file. Provided {input_data.suffix}"
        )
    if not isinstance(data, pd.DataFrame):
        raise DataSourceError(
            f"File located at {input_data} must contain a pandas DataFrame. "
            "Please provide the path to the unmodified root data directory."
        )

    return data
