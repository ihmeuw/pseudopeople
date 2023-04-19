class Keys:
    """Container for all non-dataset standard/repeated key names used in the configuration file"""

    ROW_NOISE = "row_noise"  # second layer, eg <dataset>: row_noise: {...}
    COLUMN_NOISE = "column_noise"  # second layer, eg <dataset>: column_noise: {...}
    PROBABILITY = "probability"
    CELL_PROBABILITY = "cell_probability"
    TOKEN_PROBABILITY = "token_probability"
    POSSIBLE_AGE_DIFFERENCES = "possible_age_differences"
    ZIPCODE_DIGIT_PROBABILITIES = "digit_probabilities"
