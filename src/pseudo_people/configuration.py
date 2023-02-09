from pseudo_people.entities import (
    ColumnMetadata,
    ColumnNoiseParameter,
    ColumnNoiseType,
    Form,
    RowNoiseType,
)


class ColumnNoiseConfigurationNode:
    """
    Configures levels of column noise. There are several types of column noise
    configuration including cell noise level, token noise level, etc.

    A ColumnNoiseConfigurationNode object corresponds to the configuration
    specific to a triple of form, column, and noise type.
    """

    def get_configuration(self, additional_parameter: ColumnNoiseParameter) -> float:
        """
        Access function to get this node's configuration value for a specific
        parameter.

        :param additional_parameter:
        :return:
        """
        # todo implement
        ...


class RowNoiseConfigurationNode:
    """
    Configures levels of row noise. There are two types of row noise
    configuration - omission and duplication.

    A RowNoiseConfigurationNode object corresponds to the configuration specific
    to a form and noise type pair.
    """

    def get_configuration(self) -> float:
        """
        Access function to get the configuration value for this node.

        :return:
        """
        # todo implement
        ...


class NoiseConfiguration:
    """
    The configuration of noise levels.

    There are two types of noise level configurations - row noise and column
    noise. Row noise is applied to an entire form and column noise is applied to
    a single column of a form.

    Row noise configurations can be accessed by calling
    :func: `NoiseConfiguration.get_row_noise`, while column noise configurations
    can be accessed by calling :func: `NoiseConfiguration.get_column_noise`.
    """

    def get_row_noise(
        self, form: Form, noise_type: RowNoiseType
    ) -> RowNoiseConfigurationNode:
        """
        Access method to return the row noise configuration node for the input
        form and noise type.

        :param form:
        :param noise_type:
        :return:
        """
        # todo implement
        ...

    def get_column_noise(
        self, form: Form, column_metadata: ColumnMetadata, noise_type: ColumnNoiseType
    ) -> ColumnNoiseConfigurationNode:
        """
        Access method to return the column noise configuration node for the
        input form, column, and noise type.

        :param form:
        :param column_metadata:
        :param noise_type:
        :return:
        """
        # todo implement
        ...
