from dataclasses import dataclass
from typing import Union

import pandas as pd


@dataclass
class DataFilter:
    column_name: str
    operator: str
    value: Union[str, int, pd.Timestamp]

    def to_tuple(self) -> tuple[str, str, Union[str, int, pd.Timestamp]]:
        return self.column_name, self.operator, self.value
