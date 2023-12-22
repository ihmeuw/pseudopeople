class Keys:
    """Container for all non-dataset standard/repeated key names used in the configuration file"""

    ROW_NOISE = "row_noise"  # second layer, eg <dataset>: row_noise: {...}
    COLUMN_NOISE = "column_noise"  # second layer, eg <dataset>: column_noise: {...}
    ROW_PROBABILITY = "row_probability"
    CELL_PROBABILITY = "cell_probability"
    TOKEN_PROBABILITY = "token_probability"
    POSSIBLE_AGE_DIFFERENCES = "possible_age_differences"
    ZIPCODE_DIGIT_PROBABILITIES = "digit_probabilities"
    ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18 = "row_probability_in_households_under_18"
    ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24 = (
        "row_probability_in_college_group_quarters_under_24"
    )


NO_NOISE = "no_noise"
