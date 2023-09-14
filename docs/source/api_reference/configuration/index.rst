

==============================
Working with the Configuration
==============================

The pseudopeople :ref:`configuration <configuration_main>` contains a very large number of settings.
It may be easier in some situations to interact with the configuration programmatically in Python,
rather than by creating a configuration dictionary or YAML file by hand.
For example, if you wanted to double the cell probability parameter of every noise type in every column,
or set all noise types to zero except one to isolate the effect of a specific type of noise,
it would be easier to access the defaults as a data structure in Python and modify them that way.

We currently have one function that facilitates this kind of use of the configuration, shown below.
We may add more functionality in this area in a future release of pseudopeople.

.. autofunction:: pseudopeople.get_config

.. toctree::
   :maxdepth: 2
