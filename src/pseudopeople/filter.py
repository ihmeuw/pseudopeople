from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd


@dataclass
class DataFilter:
    column_name: Optional[str]
    operator: str
    value: Union[str, int, pd.Timestamp]

    def to_tuple(self) -> tuple[Optional[str], str, Union[str, int, pd.Timestamp]]:
        return self.column_name, self.operator, self.value
