from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError


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
