.. _noise_functions_main:

=================
 Noise Functions
=================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

In order to have a realistic challenge with entity resolution, it is essential
to have noise added to the data. Pseudopeople can add two broad categories of
noise to the datasets it generates:

#. **Row-based noise:** Errors in the inclusion or exclusion of entire rows of
   data, such as duplication or omission
#. **Column-based noise:** Errors in individual data entry, such as miswriting
   or incorrectly selecting responses

Pseudopeople applies the different types of row-based and column-based noise in
the following order so as to mimic the data generation process by which a real
dataset might be subject to errors. The final column of the table, "config key,"
is the name of the noise type in the configuration dictionary used by the
dataset generation functions.

.. list-table:: Types of noise in order of application
  :widths: 1 2 5 1
  :header-rows: 1

  * - Order
    - Type of Noise
    - Description
    - Config key
  * - ---
    - **Row-based noise**
    - **Errors applied to an entire row of data, such as duplication or omission of a record**
    - ``row_noise``
  * - 1
    - Omit row
    - Omit an entire row of data
    - ``omit_row``
  * - ---
    - **Column-based noise**
    - **Errors specific to a particular column of the data, such as miswriting an address or incorrectly selecting from a list of choices**
    - ``column_noise``
  * - 2
    - Leave blank
    - Leave a field blank, such as not writing your name on the designated line
    - ``leave_blank``
  * - 3
    - Choose wrong option
    - Choose the wrong option when there are a fixed number of selections available, such as marking the "Male" box when you meant "Female"
    - ``choose_wrong_option``
