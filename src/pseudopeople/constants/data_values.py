from typing import Dict, List

import pandas as pd

from pseudopeople.constants.metadata import DatasetNames

# Targeted omission constants for do_not_respond
DO_NOT_RESPOND_BASE_PROBABILITY = 0.0024

DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_RACE: Dict[str, float] = {
    "AIAN": 0.0067,
    "Asian": -0.0286,
    "Black": 0.0306,
    "Latino": 0.0475,
    "Multiracial or Other": 0.041,
    "NHOPI": -0.0152,
    "White": -0.0188,
}

DO_NOT_RESPOND_AGE_INTERVALS: List[pd.Interval] = [
    # Intervals should include their lower bound
    pd.Interval(0, 5, closed="left"),
    pd.Interval(5, 10, closed="left"),
    pd.Interval(10, 18, closed="left"),
    pd.Interval(18, 30, closed="left"),
    pd.Interval(30, 50, closed="left"),
    pd.Interval(50, 125, closed="left"),
]

DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE: Dict[str, pd.Series] = {
    "Female": pd.Series(
        [0.0255, -0.0014, -0.0003, 0.0074, -0.0034, -0.0287],
        index=DO_NOT_RESPOND_AGE_INTERVALS,
    ),
    "Male": pd.Series(
        [0.0255, -0.0014, -0.0003, 0.0201, 0.0281, -0.0079],
        index=DO_NOT_RESPOND_AGE_INTERVALS,
    ),
}

DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY: Dict[str, float] = {
    DatasetNames.ACS: 0.0145,  # 1.45%
    DatasetNames.CPS: 0.2905,  # 29.05%
    DatasetNames.CENSUS: 0.0145,  # 1.45%
}
