from typing import Any, Union

import numpy as np
import pandas as pd

from vivarium.framework.randomness import RandomnessStream, random

from pseudopeople.entities import Form


def get_randomness_stream(form: Form, seed: int) -> RandomnessStream:
    return RandomnessStream(form.value, lambda: pd.Timestamp("2020-04-01"), seed)
