import pandas as pd

from pseudo_people.configuration import NoiseConfiguration
from pseudo_people.entities import COLUMNS_METADATA, ROW_NOISE_TYPES, Form


def noise_form(
    form: Form,
    form_data: pd.DataFrame,
    noise_config: NoiseConfiguration = None,
) -> pd.DataFrame:
    """
    Adds noise to the input form data. Row noise functions are executed which
    add or remove whole rows of data. Then column noise functions are applied.
    Each column has a set of noise functions that will get applied in a specific
    order.

    Noise levels are determined by the noise_config.
    column
    :param form:
        Form needing to be noised
    :param form_data:
        Clean data input which needs to be noised.
    :param noise_config:
        Object to configure noise levels. Default levels are used if None
    :return:
        Noised form data
    """
    for noise_type in ROW_NOISE_TYPES:
        row_configuration = noise_config.get_row_noise(form, noise_type)
        form_data = noise_type(form_data, row_configuration)

    for column_name in form_data.columns:
        column_metadata = COLUMNS_METADATA[column_name]
        for noise_type in column_metadata.noise_types:
            column_configuration = noise_config.get_column_noise(
                form, column_metadata, noise_type
            )
            form_data[column_name] = noise_type(form_data[column_name], column_configuration)

    return form_data
