from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from pseudopeople.constants import paths
from pseudopeople.interface import (
    generate_american_communities_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.schema_entities import COLUMNS


@pytest.mark.parametrize(
    "data_dir_name, noising_function, use_sample_data",
    [
        ("decennial_census_observer", generate_decennial_census, True),
        ("decennial_census_observer", generate_decennial_census, False),
        ("household_survey_observer_acs", generate_american_communities_survey, True),
        ("household_survey_observer_acs", generate_american_communities_survey, False),
        ("household_survey_observer_cps", generate_current_population_survey, True),
        ("household_survey_observer_cps", generate_current_population_survey, False),
        ("social_security_observer", generate_social_security, True),
        ("social_security_observer", generate_social_security, False),
        ("tax_w2_observer", generate_taxes_w2_and_1099, True),
        ("tax_w2_observer", generate_taxes_w2_and_1099, False),
        ("wic_observer", generate_women_infants_and_children, True),
        ("wic_observer", generate_women_infants_and_children, False),
        ("tax 1040", "todo", True),
        ("tax 1040", "todo", False),
    ],
)
def test_generate_form(
    data_dir_name: str, noising_function: Callable, use_sample_data: bool, tmpdir
):
    if noising_function == "todo":
        pytest.skip(reason=f"TODO: implement form {data_dir_name}")

    sample_data_path = list(
        (paths.SAMPLE_DATA_ROOT / data_dir_name).glob(f"{data_dir_name}*")
    )[0]

    # Load the unnoised sample data
    if sample_data_path.suffix == ".parquet":
        data = pd.read_parquet(sample_data_path)
    elif sample_data_path.suffix == ".hdf":
        data = pd.read_hdf(sample_data_path)
    else:
        raise NotImplementedError(
            f"Expected hdf or parquet but got {sample_data_path.suffix}"
        )

    # Configure if default (sample data) is used or a different root directory
    if use_sample_data:
        source = None  # will default to using sample data
    else:
        # Break sample data into two "seeds" and save to tmpdir
        outdir = tmpdir.mkdir(data_dir_name)
        suffix = sample_data_path.suffix
        split_idx = int(len(data) / 2)
        if suffix == ".parquet":
            data[:split_idx].to_parquet(outdir / f"{data_dir_name}_1{suffix}")
            data[split_idx:].to_parquet(outdir / f"{data_dir_name}_2{suffix}")
        elif suffix == ".hdf":
            data[:split_idx].to_hdf(
                outdir / f"{data_dir_name}_1{suffix}",
                "data",
                format="table",
                complib="bzip2",
                complevel=9,
            )
            data[split_idx:].to_hdf(
                outdir / f"{data_dir_name}_2{suffix}",
                "data",
                format="table",
                complib="bzip2",
                complevel=9,
            )
        else:
            raise NotImplementedError(f"Requires hdf or parquet, got {suffix}")
        source = tmpdir

    noised_data = noising_function(seed=0, source=source)
    noised_data_same_seed = noising_function(seed=0, source=source)
    noised_data_different_seed = noising_function(seed=1, source=source)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)

    # Check each column's dtype
    for col in noised_data.columns:
        expected_dtype = [c.dtype_name for c in COLUMNS if c.name == col][0]
        if expected_dtype == np.dtype(str):
            # str dtype is 'object'
            expected_dtype = np.dtype(object)
        assert noised_data[col].dtype == expected_dtype


# TODO [MIC-4000]: add test that each col to get noised actually does get noised
