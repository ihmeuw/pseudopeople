.. _glossary:

========
Glossary
========

.. glossary::

    Configuration
        The configuration is a set of parameters that will define the level for each
        noise type in a ConfigTree object.

    Datasets
        Datasets contain un-noised data will be noised and returned to the
        user with errors at a level provided by the configuration.
        The noised data will be output in format specific to the dataset type,
        eg a densus or tax form.

    Noise functions
        Functions that will be applied to datasets at a row or column level to apply
        errors or mistakes in the data to replicate realistic challenges in personal
        record linkage.