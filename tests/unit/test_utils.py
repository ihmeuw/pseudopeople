import numpy as np
import pandas as pd

from pseudopeople.schema_entities import DtypeNames
from pseudopeople.utilities import cleanse_object_columns


def test_clenase_object_columns():
    # This tests that object columns return only strings.
    # This is to handle dtype issues we were having with int/float/strings in
    # age, wages, and po box columns.
    s = pd.Series([np.nan, 1, "2.0", 3.0, 4.0, "5.0", np.nan])
    t = cleanse_object_columns(s)

    assert s.dtype.name == t.dtype.name
    assert t.dtype.name == DtypeNames.OBJECT

    expected = pd.Series([np.nan, "1", "2", "3", "4", "5", np.nan])

    pd.testing.assert_series_equal(t, expected)