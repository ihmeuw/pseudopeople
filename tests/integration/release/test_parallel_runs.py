import pandas as pd
import time

from pytest_check import check
from collections.abc import Callable
from pathlib import Path
from typing import Literal


def test_all_passing() -> None:
    '''Test which passes no matter what parameters are called.'''
    pass


def test_some_failing(
    dataset_params: tuple[
        str,
        Callable[..., pd.DataFrame],
        Path | str | None,
        int | None,
        str | None,
        Literal["pandas", "dask"],
        str,
    ],    
) -> None:
    '''Test which fails for some but not all parameters.'''
    dataset_name, _, _, _, _, _, _ = dataset_params
    if dataset_name == 'census':
        with check:
            assert False


def test_cancellation(
    dataset_params: tuple[
        str,
        Callable[..., pd.DataFrame],
        Path | str | None,
        int | None,
        str | None,
        Literal["pandas", "dask"],
        str,
    ],    
) -> None:
    '''Test which will get cancelled for tax_1040 when job is called by sbatch
    with a 1 minute srun.'''
    dataset_name, _, _, _, _, _, _ = dataset_params
    if dataset_name == 'tax_1040':
        time.sleep(60 * 5) # 5 minutes
