from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.utilities import PANDAS_ENGINE, DataFrame, Engine


def load_standard_dataset(
    data_path: Path,
    user_filters: List[Tuple],
    engine: Engine = PANDAS_ENGINE,
    is_file: bool = True,
) -> DataFrame:
    if is_file and data_path.suffix != ".parquet":
        raise DataSourceError(
            f"Source path must be a .parquet file. Provided {data_path.suffix}"
        )

    if engine == PANDAS_ENGINE:
        if len(user_filters) == 0:
            # pyarrow.parquet.read_table doesn't accept an empty list
            user_filters = None
        data = pq.read_table(data_path, filters=user_filters).to_pandas()

        # TODO: The index in our simulated population files is never meaningful.
        # For some reason, the 1040 dataset is currently saved with a non-RangeIndex
        # in the large data, and all datasets have a non-RangeIndex in the sample data.
        # If we don't drop these here, our index can have duplicates when we load multiple
        # shards at once. Having duplicates in the index breaks much of
        # our noising logic.
        data.reset_index(drop=True, inplace=True)
    else:
        # Dask
        import dask.dataframe as dd

        data = dd.read_parquet(str(data_path), filters=user_filters)
        # See TODO above.
        data = data.reset_index(drop=True)

    if not isinstance(data, engine.dataframe_class):
        raise DataSourceError(
            f"File located at {data_path} must contain a DataFrame. "
            "Please provide the path to the unmodified root data directory."
        )

    return data
