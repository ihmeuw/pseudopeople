.. _dataset_generation_functions:

============================
Dataset Generation Functions
============================

Each of the following functions generates one of the simulated datasets
documented on the :ref:`Datasets page <datasets_main>`.
For example, :func:`pseudopeople.generate_decennial_census` generates the Decennial Census dataset.

All of the dataset generation functions have the same (optional) parameters.
Notable parameters include:

    - a `source` path to the root directory of :ref:`pseudopeople simulated population data <simulated_populations_main>` (defaults to using the sample population included with pseudopeople).
    - a `config` path to a YAML file, a Python dictionary, or the special value :code:`pseudopeople.NO_NOISE`, to :ref:`override the default configuration <configuration_main>`.
    - a `year` (defaults to 2020).

For applied examples of using these functions, see the :ref:`Quickstart <quickstart>` and :ref:`tutorials <tutorial_main>`.

.. automodule:: pseudopeople
   :imported-members:
   :exclude-members: get_config

.. toctree::
   :maxdepth: 2
