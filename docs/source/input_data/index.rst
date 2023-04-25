.. _input_data_main:

==========
Input Data
==========

pseudopeople leverages the power of the `Vivarium
<https://vivarium.readthedocs.io/en/latest/>`_ microsimulation platform to
incorporate real, publicly accessible data about the US population. The input
data for pseudopeople is the output of a Vivarium simulation and must be in a
specific format for the dataset generating functions to work. There are
currently three collections of pseudopeople input data:

- Sample data (a fictional population of ~10,000 simulants living in Anytown, US, included with the pseudopeople package)
- Rhode Island (a fictional population of ~1,000,000 simulants living in a simulated state of Rhode Island)
- United States (a fictional population of ~330,000,000 simulants living throughout a simulated United States)

When generating a dataset, pseudopeople uses the included sample data by default
unless an explicit path to another directory containing pseudopeople input data
is specified.

Accessing the large-scale input data
------------------------------------

To gain access to the larger-scale input data (i.e., Rhode Island and United States),
follow these steps:

#. Log in to `GitHub <https://github.com/>`_ (you must first create a GitHub account if you don't have one).
#. Open a new `Data access request <https://github.com/ihmeuw/pseudopeople/issues/new?assignees=&labels=&template=data_access_request.yml>`_ using the template under the Issues tab on pseudopeople's GitHub page.
#. Fill out the information on the access request form to tell us about your project. You can simply put "Data access request" in the title field.
#. We will get back to you after we receive your request!

Using pseudopeople input data
-----------------------------

Once you've downloaded the large-scale input data (either Rhode Island or United
States), unzip the contents to the desired location on your computer.

.. important::

  Do not modify the contents of the directory containing the unzipped input
  data! Modifications to the pseudopeople input data may cause the dataset
  generating functions to fail.

Once you've unzipped the input data, you can pass the directory path to the
:ref:`dataset generation functions <dataset_generation_functions>` to generate large-scale datasets!
