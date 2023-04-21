.. _noise_main:

======
 Noise
======

.. contents::
   :depth: 2
   :local:
   :backlinks: entry

In order to have a realistic challenge with entity resolution, it is essential
to add noise to the simulated data. "Noise" refers to various types of errors
introduced into the data and may also be called "corruption" or "distortion." By
default, Pseudopeople applies noise to the simulated datasets using some
reasonable settings. If desired, the user can change the noise settings through
the configuration system---see the Configuration section for details.

.. todo::

  Add link to Configuration section once it exists.

Categories of Noise
-------------------

Pseudopeople can add two broad categories of noise to the datasets it generates:

#. **Row-based noise:** Errors in the inclusion or exclusion of entire rows of
   data, such as duplication or omission
#. **Column-based noise:** Errors in data entry for individual fields within a
   row, such as miswriting or incorrectly selecting responses

Each type of row-based noise operates on the entire dataset (selecting rows to
include or exclude), while each type of column-based noise operates on one
column of data at a time (selecting cells within that column to noise).
Currently, errors added in different columns are independent of each other.

Available Noise Types
---------------------

Pseudopeople applies the different types of row-based and column-based noise in
the following order to mimic the data generation process by which a real dataset
might be corrupted. The table lists all the available noise types, but not every
type of noise will necessarily be applied to every dataset or every column. The
configuration determines which noise types are actually used. The "Noise Type"
column shows the name of the noise type in the configuration system.

.. list-table:: Types of noise in order of application
  :widths: 1 2 5 1
  :header-rows: 1

  * - Noise Type
    - Description
    - Example
    - Order
  * - ``row_noise``
    - **Row-based noise**
    - ---
    - ---
  * - ``omit_row``
    - Omit a row
    - Neglecting to file a tax form on time
    - 1
  * - ``column_noise``
    - **Column-based noise**
    - ---
    - ---
  * - ``leave_blank``
    - Leave a field blank
    - Forgetting to write your name on the designated line
    - 2
  * - ``choose_wrong_option``
    - Choose the wrong option from a fixed set of options
    - Marking the "Male" box when you meant "Female"
    - 3

.. todo::

  Fill in the remaining noise functions in the above table.

Default Noise Types for Each Column
-----------------------------------

Noise Function Details
----------------------

.. toctree::
  :maxdepth: 2

  row_noise
  column_noise
