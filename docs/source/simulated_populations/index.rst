.. _simulated_populations_main:

=====================
Simulated populations
=====================

.. _Vivarium: https://vivarium.readthedocs.io/en/latest/

pseudopeople generates multiple :ref:`datasets <datasets_main>` about a
simulated population which can be specified by the user when calling the
:ref:`dataset generation functions <dataset_generation_functions>`. There are
currently three simulated populations available for generating datasets with
pseudopeople:

- **Sample population** (a fictional population of ~10,000 simulants living in the fictional Anytown, WA, included with the pseudopeople package)
- **Rhode Island** (a fictional population of ~1,000,000 simulants living in a simulated state of Rhode Island)
- **United States** (a fictional population of ~330,000,000 simulants living throughout a simulated United States)

When generating a dataset, pseudopeople uses the included sample population by
default unless an explicit path to another directory containing simulated
population data is specified. See the sections below for more information about
accessing and using the larger simulated populations.

.. contents::
  :local:


**Note:** The simulated population data used by pseudopeople is the output of a
Vivarium_ microsimulation and must be in a specific format for the dataset
generation functions to work. Vivarium uses real, publicly available data to
stochastically simulate multiple decades of population dynamics such as
fertility, mortality, migration, and employment. Then pseudopeople takes the
simulated population data output by Vivarium and simulates the data collection
process with user-configurable :ref:`noise <noise_main>` added to the resulting
datasets.

..
  The entire simulation process can be visualized as follows.

  [[Insert image here]]

Accessing the large-scale simulated populations
-----------------------------------------------

To gain access to the larger-scale simulated populations (i.e., Rhode Island and
United States), follow these steps:

#. Log in to `GitHub <https://github.com/>`_ (you must first create a GitHub account if you don't have one).
#. Open a new `Data access request <https://github.com/ihmeuw/pseudopeople/issues/new?assignees=&labels=&template=data_access_request.yml>`_ using the template under the `Issues tab <https://github.com/ihmeuw/pseudopeople/issues>`_ on pseudopeople's GitHub page.
#. Fill out the information on the access request form to tell us about your project. You can simply put "Data access request" in the title field.
#. We will get back to you after we receive your request!

Validating the simulated population data
----------------------------------------

.. _Checksums: https://en.wikipedia.org/wiki/Checksum

Checksums_ can be used to validate that you've successfully
downloaded the correct and uncorrupted zip file.
The following table provides the SHA-256 checksum for the larger-scale simulated population zip files:

.. list-table:: SHA-256 checksums
  :header-rows: 1

  * - Location
    - File
    - SHA-256 checksum
  * - Rhode Island
    - pseudopeople_simulated_population_ri_2_0_1.zip
    - fadcbf40c87217f77f36f2c684a6a568460a1215696bc2f8a0c2069a00cdc78c
  * - US
    - pseudopeople_simulated_population_usa_2_0_0.zip
    - 0025978196c2a84c1df502e857bec35a84c25092fbfb6b143c0b8ff30dea5eed
  * - Rhode Island
    - pseudopeople_simulated_population_ri_2_0_0.zip
    - bfec148c947096b44201a7961a1b38f63961cd820578f10a479f623d8d79f0d1
  * - US
    - pseudopeople_simulated_population_usa_1_0_0.zip
    - 9462cc60b333fb2a3d16554a9e59b5428a81a2b1d2c34ed383883d7b68d2f89f
  * - Rhode Island
    - pseudopeople_simulated_population_ri_1_0_0.zip
    - d3f1ccdfbfca8b53254c4ceeb18afe17c3d3b3fe02f56cc20d1254f818c39435

If the SHA-256 checksum that
you generate for the downloaded file matches the value provided above, you can
be sure you downloaded the file successfully.

Possibly the simplest way to verify checksums is to generate the value using the
terminal/cmd command below (be sure to replace `PATH/TO/ZIP`  with the actual path
to the zip you downloaded) and visually compare the result to the
values provided above. Note that if even the first few and last few characters
match then it is very likely the entire string matches.

Linux:

.. code-block:: console

  $ sha256sum PATH/TO/ZIP

Mac:

.. code-block:: console

  $ shasum -a 256 PATH/TO/ZIP

Windows:

.. code-block:: console

  $ CertUtil -hashfile PATH/TO/ZIP SHA256

.. note::

  Generating the checksum can take a long time for larger files, e.g. several
  minutes for the Rhode Island dataset and ~1 hour for the United States dataset.

If the generated checksum does not match the one provided in the table above,
please try re-downloading the dataset.

If after downloading the file a second time the checksums still do not match,
please open a `Bug report <https://github.com/ihmeuw/pseudopeople/issues/new?assignees=&labels=&template=bug_report.yml>`_
using the template under the `Issues tab <https://github.com/ihmeuw/pseudopeople/issues>`_
on pseudopeople's GitHub page.

Using the simulated population data
-----------------------------------

Once you've downloaded the large-scale simulated population (either Rhode Island
or United States), unzip the contents to the desired location on your computer.

.. important::

  Do not modify the contents of the directory containing the unzipped simulated
  population data! Modifications to the pseudopeople simulated population data may cause the
  dataset generation functions to fail.

Once you've unzipped the simulated population data, you can pass the directory
path to the :code:`source` parameter of the :ref:`dataset generation functions
<dataset_generation_functions>` to generate large-scale datasets!
