.. _noise_main:

======
 Noise
======

.. contents::
   :depth: 2
   :local:
   :backlinks: entry

In order to have a realistic challenge with entity resolution, it is essential
to add noise to the data. Noise can also be called "corruption" or "distortion."
This page describes the types of noise included in Pseudopeople.

Categories of noise
-------------------

Pseudopeople can add two broad categories of noise to the datasets it generates:

#. **Row-based noise:** Errors in the inclusion or exclusion of entire rows of
   data, such as duplication or omission
#. **Column-based noise:** Errors in data entry for individual fields within a
   row, such as miswriting or incorrectly selecting responses

Each type of row-based noise operates on the entire dataset (selecting rows to
include or exclude), while each type of column-based noise operates on one
column of data at a time (selecting cells within that column to noise).
Pseudopeople comes with a default noise configuration for the provided datasets,
and the defaults can be overridden by the user. See the Configuration section
for details.

.. todo::

  Add link to Configuration section once it exists.

Available noise types
---------------------

Pseudopeople applies the different types of row-based and column-based noise in
the following order to mimic the data generation process by which a real dataset
might be corrupted. The table lists all the available noise types, but not every
type of noise will necessarily be applied to every dataset or every column. The
configuration determines which noise types are actually used. The "Config Key"
column shows the name of the noise type in the configuration.

.. list-table:: Types of noise in order of application
  :widths: 1 2 5 1
  :header-rows: 1

  * - Config Key
    - Noise Type
    - Description
    - Order
  * - ``row_noise``
    - **Row-based noise**
    - ---
    - ---
  * - ``omit_row``
    - Omit a row
    - Omit an entire randomly selected row of data
    - 1
  * - ``column_noise``
    - **Column-based noise**
    - ---
    - ---
  * - ``leave_blank``
    - Leave a field blank
    - For example, forgetting to write your name on the designated line
    - 2
  * - ``choose_wrong_option``
    - Choose the wrong option from a fixed set of options
    - For example, marking the "Male" box when you meant "Female"
    - 3

.. todo::

  Fill in the remaining noise functions in the above table.

Default noise types for each column
-----------------------------------

Noise function details
----------------------
