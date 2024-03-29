.. _glossary:

========
Glossary
========

.. glossary::

    Configuration
        Settings that allow you to customize the noise present in the datasets
        generated by pseudopeople.
        Noise is configurable at a very fine-grained level, with settings
        specific to the dataset, noise type, and column (where applicable).

    Datasets
        The types of data that can be simulated with pseudopeople, each of which is
        the simulated analog of a "real world" database from a census, survey, or administrative source.
        For example, pseudopeople's American Community Survey (ACS) dataset is analogous to the
        data that would be collected by that survey in real life.

    Entity resolution (ER)
        The task of identifying the unique entities associated with a set of records,
        where multiple records may refer to the same entity.
        Also called "record linkage," among other names.

    Noise
        Errors introduced to a pseudopeople dataset.
        These simulate data errors that would be found in real-life survey
        and administrative data.

    Noise types
        The types of error that can be introduced to a pseudopeople dataset.
        Each one simulates a specific type of mistake or inaccuracy that could occur in
        a real-life data collection or generation process.
        For example, one of the noise types in pseudopeople is
        a simulant choosing the wrong option from a list of choices on a form.

    Probabilistic record linkage (PRL)
        Entity resolution ("record linkage") methods that internally use probabilities of
        some kind to represent uncertainty about which records belong to which entities.

    Record linkage
        Another term for entity resolution.

    Simulant
        A simulated person represented in a pseudopeople-generated dataset.