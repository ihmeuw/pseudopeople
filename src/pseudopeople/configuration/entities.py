class Keys:
    """Container for all non-dataset standard/repeated key names used in the configuration file"""

    ROW_NOISE = "row_noise"  # second layer, eg <dataset>: row_noise: {...}
    COLUMN_NOISE = "column_noise"  # second layer, eg <dataset>: column_noise: {...}
    ROW_PROBABILITY = "row_probability"
    CELL_PROBABILITY = "cell_probability"
    TOKEN_PROBABILITY = "token_probability"
    POSSIBLE_AGE_DIFFERENCES = "possible_age_differences"
    ZIPCODE_DIGIT_PROBABILITIES = "digit_probabilities"
    IN_HOUSEHOLDS_UNDER_18 = "in_households_under_18"
    IN_HOUSEHOLDS_18_TO_23 = "in_households_18_to_23"
    IN_GROUP_QUARTERS_UNDER_24 = "in_group_quarters_under_24"


NO_NOISE = "no_noise"
