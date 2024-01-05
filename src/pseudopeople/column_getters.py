from typing import List

from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS


def copy_from_household_member_column_getter(column_name) -> List[str]:
    return [COPY_HOUSEHOLD_MEMBER_COLS[column_name]]
