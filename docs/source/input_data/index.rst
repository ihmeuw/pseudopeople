.. _input_data_main:

==========
Input Data
==========

pseudopeople leverages the power of the `Vivarium
<https://vivarium.readthedocs.io/en/latest/>`_ microsimulation platform to
incorporate real, publicly accessible data about the US population. The input
data for pseudopeople is the output of a Vivarium simulation and must be in a
specific format for the :ref:`dataset generation functions
<dataset_generation_functions>` to work. There are currently three collections
of pseudopeople input data:

- **Sample data** (a fictional population of ~10,000 simulants living in Anytown, US, included with the pseudopeople package)
- **Rhode Island** (a fictional population of ~1,000,000 simulants living in a simulated state of Rhode Island)
- **United States** (a fictional population of ~330,000,000 simulants living throughout a simulated United States)

When generating a dataset, pseudopeople uses the included sample data by default
unless an explicit path to another directory containing pseudopeople input data
is specified.

Accessing the large-scale input data
------------------------------------

To gain access to the larger-scale input data (i.e., Rhode Island and United States),
follow these steps:

#. Log in to `GitHub <https://github.com/>`_ (you must first create a GitHub account if you don't have one).
#. Open a new `Data access request <https://github.com/ihmeuw/pseudopeople/issues/new?assignees=&labels=&template=data_access_request.yml>`_ using the template under the `Issues tab <https://github.com/ihmeuw/pseudopeople/issues>`_ on pseudopeople's GitHub page.
#. Fill out the information on the access request form to tell us about your project. You can simply put "Data access request" in the title field.
#. We will get back to you after we receive your request!

Validating pseudopeople input data
----------------------------------

The following table provides the SHA-256 checksum for the larger-scale input
data zip files:

.. list-table:: SHA-256 checksums
  :header-rows: 1

  * - Location
    - File
    - SHA-256 checksum
  * - US
    - pseudopeople_input_data_usa_1_0_0.zip
    - 9462cc60b333fb2a3d16554a9e59b5428a81a2b1d2c34ed383883d7b68d2f89f
  * - Rhode Island
    - pseudopeople_input_data_ri_1_0_0.zip
    - d3f1ccdfbfca8b53254c4ceeb18afe17c3d3b3fe02f56cc20d1254f818c39435

These SHA-256 checksums can be used to validate that you've successfully
downloaded the correct (and uncorrupt) zip file; if the SHA-256 checksum that
you generate for the downloaded file matches the value provided in the table
above, you can be sure the download was successful. There are many tools to
generate SHA-256 checksums, including the terminal/cmd commands below:

Linux:

.. code-block:: console

  $ sha256sum <filename>

Mac:

.. code-block:: console

  $ shasum -a 256 <filename>

Windows:

.. code-block:: console

  $ CertUtil -hashfile <filename> SHA256

Using pseudopeople input data
-----------------------------

Once you've downloaded the large-scale input data (either Rhode Island or United
States), unzip the contents to the desired location on your computer.

.. important::

  Do not modify the contents of the directory containing the unzipped input
  data! Modifications to the pseudopeople input data may cause the dataset
  generation functions to fail.

Once you've unzipped the input data, you can pass the directory path to the
:ref:`dataset generation functions <dataset_generation_functions>` to generate large-scale datasets!
