from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import yaml
from vivarium.framework.configuration import ConfigTree
from vivarium.framework.randomness import RandomnessStream, random

from pseudopeople.schema_entities import Form


def get_randomness_stream(form: Form, seed: int) -> RandomnessStream:
    return RandomnessStream(form.value, lambda: pd.Timestamp("2020-04-01"), seed)


def get_configuration(user_configuration: Union[Path, str, Dict] = None) -> ConfigTree:
    """
    Gets a noising configuration ConfigTree, optionally overridden by a user-provided YAML.

    :param user_configuration: A path to the YAML file or a dictionary defining user overrides for the defaults
    :return: a ConfigTree object of the noising configuration
    """
    import pseudopeople

    default_config_layers = [
        "base",
        "user",
    ]
    noising_configuration = ConfigTree(
        data=Path(pseudopeople.__file__).resolve().parent / "default_configuration.yaml",
        layers=default_config_layers,
    )
    if user_configuration:
        if isinstance(user_configuration, (Path, str)):
            with open(user_configuration, "r") as f:
                user_configuration = yaml.full_load(f)
        user_configuration = format_user_configuration(
            user_configuration, noising_configuration
        )
        noising_configuration.update(user_configuration, layer="user")

    validate_noising_configuration(noising_configuration)

    return noising_configuration


def format_user_configuration(user_dict: Dict, default_config) -> Dict:
    """Formats the user's configuration file as necessary so it can properly
    update noising configuration to be used
    """
    user_dict = _format_age_miswriting_perturbations(user_dict, default_config)

    return user_dict


def _format_age_miswriting_perturbations(user_dict: Dict, default_config: ConfigTree) -> Dict:
    # Format any age perturbation lists as a dictionary with uniform probabilites
    for form in user_dict:
        user_perturbations = (
            user_dict[form]
            .get("age", {})
            .get("age_miswriting", {})
            .get("possible_perturbations", {})
        )
        if not user_perturbations:
            continue
        formatted = {}
        default_perturbations = default_config[form]["age"]["age_miswriting"][
            "possible_perturbations"
        ]
        # Replace default configuration with 0 probabilities
        for perturbation in default_perturbations:
            formatted[perturbation] = 0
        if isinstance(user_perturbations, list):
            # Add user perturbations with uniform probabilities
            uniform_prob = 1 / len(user_perturbations)
            for perturbation in user_perturbations:
                formatted[perturbation] = uniform_prob
        elif isinstance(user_perturbations, dict):
            for perturbation, prob in user_perturbations.items():
                formatted[perturbation] = prob
        else:
            raise NotImplementedError(
                "age.age_miswriting.possible_perturbations can only be a list or dict, "
                f"received type {type(user_perturbations)}"
            )
        user_dict[form]["age"]["age_miswriting"]["possible_perturbations"] = formatted

    return user_dict


def validate_noising_configuration(config: ConfigTree) -> None:
    """Perform various validation checks on the final noising ConfigTree object"""
    _validate_age_miswriting(config)


def _validate_age_miswriting(config: ConfigTree) -> None:
    possible_perturbations = extract_values(config, "possible_perturbations")
    for form_perturbations in possible_perturbations:
        form_perturbations_dict = form_perturbations.to_dict()
        if 0 in form_perturbations_dict:
            # TODO: Find a way to report specific location in config file
            raise ValueError("Cannot include 0 in age_miswriting.possible_perturbations")
        if sum(form_perturbations_dict.values()) != 1:
            raise ValueError(
                "The provided possible_perturbation probabilities must sum to 1 but they "
                f"currently sum to {sum(form_perturbations_dict.values())}: {form_perturbations_dict}",
            )


def extract_values(config: Union[ConfigTree, Dict], key: str):
    """Extract values with a specific key from a dict or configtree"""
    results = []
    for k, v in config.items():
        if k == key:
            results.append(v)
        if isinstance(v, (dict, ConfigTree)):
            for result in extract_values(v, key):
                results.append(result)

    return results


def vectorized_choice(
    options: Union[list, pd.Series],
    n_to_choose: int,
    randomness_stream: RandomnessStream = None,
    weights: Union[list, pd.Series] = None,
    additional_key: Any = None,
    random_seed: int = None,
):
    """
    Function that takes a list of options and uses Vivarium common random numbers framework to make a given number
    of random choice selections.

    :param options: List and series of possible values to choose
    :param n_to_choose: Number of choices to make, the length of the returned array of values
    :param randomness_stream: RandomnessStream being used for Vivarium's CRN framework
    :param weights: List or series containing weights for each options
    :param additional_key: Key to pass to randomness_stream
    :param random_seed: Seed to pass to randomness_stream.
    Note additional_key and random_seed are used to make calls using a RandomnessStream unique

    returns: ndarray
    """
    if not randomness_stream and (additional_key == None and random_seed == None):
        raise RuntimeError(
            "An additional_key and a random_seed are required in 'vectorized_choice'"
            + "if no RandomnessStream is passed in"
        )
    if weights is None:
        n = len(options)
        weights = np.ones(n) / n
    if isinstance(weights, list):
        weights = np.array(weights)
    # for each of n_to_choose, sample uniformly between 0 and 1
    index = pd.Index(np.arange(n_to_choose))
    if randomness_stream is None:
        # Generate an additional_key on-the-fly and use that in randomness.random
        additional_key = f"{additional_key}_{random_seed}"
        probs = random(str(additional_key), index)
    else:
        probs = randomness_stream.get_draw(index, additional_key=additional_key)

    # build cdf based on weights
    pmf = weights / weights.sum()
    cdf = np.cumsum(pmf)

    # for each p_i in probs, count how many elements of cdf for which p_i >= cdf_i
    chosen_indices = np.searchsorted(cdf, probs, side="right")
    return np.take(options, chosen_indices)


def get_index_to_noise(
    column: pd.Series,
    noise_level: float,
    randomness_stream: RandomnessStream,
    additional_key: Any,
) -> pd.Index:
    """
    Function that takes a series and returns a pd.Index that chosen by Vivarium Common Random Number to be noised.
    """

    # Get rows to noise
    not_empty_idx = column.index[(column != "") & (column.notna())]
    to_noise_idx = randomness_stream.filter_for_probability(
        not_empty_idx,
        probability=noise_level,
        additional_key=additional_key,
    )

    return to_noise_idx
