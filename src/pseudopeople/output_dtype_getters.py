import numpy as np

from pseudopeople.dtypes import DtypeNames


def output_dtype_getter_leave_blank(dtype: np.dtype) -> np.dtype:
    # Make sure the dtype is nullable
    if "int" in dtype.name:
        return "float"

    return dtype


def output_dtype_getter_always_string(_: np.dtype) -> np.dtype:
    return DtypeNames.OBJECT
