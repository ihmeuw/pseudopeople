from typing import Dict

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

DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE: pd.DataFrame = pd.DataFrame(
    [
        ["Female", pd.Interval(0, 4), 0.0255],
        ["Female", pd.Interval(5, 9), -0.0014],
        ["Female", pd.Interval(10, 17), -0.0003],
        ["Female", pd.Interval(18, 29), 0.0074],
        ["Female", pd.Interval(30, 49), -0.0034],
        ["Female", pd.Interval(50, 125), -0.0287],
        ["Male", pd.Interval(0, 4), 0.0255],
        ["Male", pd.Interval(5, 9), -0.0014],
        ["Male", pd.Interval(10, 17), -0.0003],
        ["Male", pd.Interval(18, 29), 0.0201],
        ["Male", pd.Interval(30, 49), 0.0281],
        ["Male", pd.Interval(50, 125), -0.0079],
    ],
    columns=["sex", "age_interval", "probability"],
)

DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY: Dict[str, float] = {
    DatasetNames.ACS: 0.0145,  # 1.45%
    DatasetNames.CPS: 0.2905,  # 29.05%
    DatasetNames.CENSUS: 0.0145,  # 1.45%
}
