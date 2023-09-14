.. _configuration_main:

=============
Configuration
=============

You can customize the noise in the datsets the pseudopeople package generates.
This allows you to explore different scenarios and see how sensitive entity resolution methods
are to the types and levels of noise present in their input data.

Overriding defaults
-------------------

Noise is configurable at a very fine-grained level.
It can be customized separately for each **dataset** and **noise type**.
:ref:`Column-based noise types <categories_of_noise>` can additionally have different settings for each **column**.

Due to this fine-grained control, there are a very large number of settings.
**It is not necessary to configure everything.**
pseudopeople includes reasonable default noise settings
and your configuration can override as few or as many of the default values as you like.
You can also pass the special value :code:`pseudopeople.NO_NOISE`, which prevents all configurable noise types
from occurring at all.

To learn more about the default settings, see :ref:`Noise Type Details <noise_type_details>`.
You can access the defaults from your Python code by calling the :func:`pseudopeople.get_config` function.

.. _configuration_structure:

Configuration structure
-----------------------

Configuration can be supplied as a nested Python dictionary, or as a YAML file.
In either case, the structure is the same:

* The top-level keys are the datasets.
* Within each of these are keys for the :ref:`categories of noise <categories_of_noise>`: row-based and column-based.
* **For column-based noise-only**, the next layer of keys is for the columns in the dataset.
* Nested within these are keys for the individual :ref:`noise types <available_noise_types>`.
* Finally, each noise type has parameters.

As an example, say we wanted to change the cell probability parameter (which is the probability of a cell being wrong)
of the :ref:`Choose the wrong option <choose_the_wrong_option>` noise type, for the sex column of the Decennial Census dataset.
Here are the configurations to do this in Python and YAML, respectively:

.. code-block:: python

    config = {
        'decennial_census': { # Dataset
            'column_noise': { # "Choose the wrong option" is in the column-based noise category
                'sex': { # Column
                    'choose_wrong_option': { # Noise type
                        'cell_probability': 0.05, # Parameter (and value)
                    },
                },
            },
        },
    }

.. code-block:: yaml

    decennial_census: # Dataset
        column_noise: # "Choose the wrong option" is in the column-based noise category
            sex: # Column
                choose_wrong_option: # Noise type
                    cell_probability: 0.05 # Parameter (and value)

Row-based noise is similar, except that there is no key to specify the column, since it is not column-specific.
For example to change the probability of :ref:`nonresponse <do_not_respond>` in the Decennial Census, the configuration would be:

.. code-block:: python

    config = {
        'decennial_census': { # Dataset
            'row_noise': { # "Omit a row" is in the row-based noise category
                'do_not_respond': { # Noise type
                    'row_probability': 0.05, # Parameter (and value)
                },
            },
        },
    }

.. code-block:: yaml

    decennial_census: # Dataset
        row_noise: # "Omit a row" is in the row-based noise category
            do_not_respond: # Noise type
                row_probability: 0.05 # Parameter (and value)

How to pass configuration to pseudopeople
-----------------------------------------

Each of pseudopeople's :ref:`dataset generation functions <dataset_generation_functions>` takes a :code:`config`
argument.
This argument can be passed either a Python dictionary, the path to a YAML file, or the special value
:code:`pseudopeople.NO_NOISE`, which prevents all configurable noise types from occurring at all.

Configurable parameters
-----------------------

The noise types that can be configured, and the parameters of each,
are listed in the :ref:`Noise Type Details <noise_type_details>` section.
