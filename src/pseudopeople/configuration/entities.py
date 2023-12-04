class Keys:
    """Container for all non-dataset standard/repeated key names used in the configuration file"""

    ROW_NOISE = "row_noise"  # second layer, eg <dataset>: row_noise: {...}
    COLUMN_NOISE = "column_noise"  # second layer, eg <dataset>: column_noise: {...}
    ROW_PROBABILITY = "row_probability"
    CELL_PROBABILITY = "cell_probability"
    TOKEN_PROBABILITY = "token_probability"
    POSSIBLE_AGE_DIFFERENCES = "possible_age_differences"
    ZIPCODE_DIGIT_PROBABILITIES = "digit_probabilities"
    GUARDIAN_BASED_DUPLCATION_PROBABILITIES = "guardian_based_duplication_probabilities"
    UNDER_18_IN_HOUSEHOLDS = "under_18_in_households"
    IN_HOUSEHOLDS_18_TO_23 = "in_households_18_to_23"
    UNDER_24_IN_GROUP_QUARTERS = "under_24_in_group_quarters"


NO_NOISE = "no_noise"
