.. _noise_main:

======
 Noise
======

.. contents::
   :depth: 2
   :local:
   :backlinks: none

In order to have a realistic challenge with entity resolution, it is essential
to add noise to the data. Pseudopeople can add two broad categories of noise to
the datasets it generates:

#. **Row-based noise:** Errors in the inclusion or exclusion of entire rows of
   data, such as duplication or omission of a record
#. **Column-based noise:** Errors in data entry for individual fields within a
   record, such as miswriting or incorrectly selecting responses

Pseudopeople applies the different types of row-based and column-based noise in
the following order to mimic the data generation process by which a real dataset
might be corrupted. The value in the "Config Key" column is the name of
the noise type in the configuration used to customize the noise settings.

.. list-table:: Types of noise in order of application
  :widths: 1 1 2 5
  :header-rows: 1

  * - Config Key
    - Order
    - Noise Type
    - Description
  * - ``row_noise``
    - ---
    - **Row-based noise**
    - ---
  * - ``omit_row``
    - 1
    - Omit a row
    - Omit an entire randomly selected row of data
  * - ``column_noise``
    - ---
    - **Column-based noise**
    - ---
  * - ``leave_blank``
    - 2
    - Leave a field blank
    - For example, not writing your name on the designated line
  * - ``choose_wrong_option``
    - 3
    - Choose the wrong option (applies only to multiple choice fields)
    - For example, marking the "Male" box when you meant "Female"

.. todo::

  Fill in the remaining noise functions in the above table.
