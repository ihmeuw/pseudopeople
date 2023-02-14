import pandas as pd

from pseudo_people.configuration import NoiseConfiguration
from pseudo_people.entities import DEFAULT_CONFIGURATION, NOISE_TYPES, Form
from pseudo_people.entity_types import ColumnNoiseType, RowNoiseType


def noise_form(
    form: Form,
    form_data: pd.DataFrame,
    noise_config: NoiseConfiguration = None,
) -> pd.DataFrame:
    """
    Adds noise to the input form data. Noise functions are executed in the order
    defined by :py:const: `.NOISE_TYPES`. Row noise functions are applied to the
    whole DataFrame. Column noise functions will be applied to each column that
    is pertinent to it.

    Noise levels are determined by the noise_config.
    :param form:
        Form needing to be noised
    :param form_data:
        Clean data input which needs to be noised.
    :param noise_config:
        Object to configure noise levels. Default levels are used if None
    :return:
        Noised form data
    """

    if noise_config is None:
        noise_config = DEFAULT_CONFIGURATION

    for noise_type in NOISE_TYPES:
        if isinstance(noise_type, RowNoiseType):
            # Apply row noise
            row_configuration = noise_config.get_row_noise(form, noise_type)
            form_data = noise_type(form_data, row_configuration)

        elif isinstance(noise_type, ColumnNoiseType):
            # Apply column noise to each column as appropriate
            for column in form_data.columns:
                if column not in noise_type.columns:
                    continue

                column_configuration = noise_config.get_column_noise(form, column, noise_type)
                form_data[column] = noise_type(form_data[column], column_configuration)
        else:
            raise TypeError(
                f"Invalid noise type. Allowed types are {RowNoiseType} and "
                f"{ColumnNoiseType}. Provided {type(noise_type)}."
            )

    return form_data
