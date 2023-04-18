from pseudopeople.configuration.generator import get_configuration


def get_config(form_name: str, user_configuration: dict) -> None:
    """
    Function that displays the configuration for the user
    :param form_name: Name of form to lookup in configuration.  Providing this argument returns the configuration for
      this specific form and no other forms in the configuration.
    :param user_configuration: Dictionary of configuration values the user wishes to manually override.
    """

    config = get_configuration(user_configuration)
    if form_name:
        config = config[form_name]
    print(config)

    return None
