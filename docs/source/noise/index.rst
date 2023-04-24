.. _noise_main:

======
 Noise
======

In order to have a realistic challenge with entity resolution, it is essential
to add noise to the simulated data. "Noise" refers to various types of errors
introduced into the data and may also be called "corruption" or "distortion." By
default, Pseudopeople applies noise to the simulated datasets using some
reasonable settings. If desired, the user can change the noise settings through
the configuration system---see the :ref:`Configuration section <configuration_main>` 
for details.

.. contents::
   :depth: 2
   :local:
   :backlinks: entry

.. todo::

  Add link to Configuration section once it exists.

Categories of Noise
-------------------

Pseudopeople can add two broad categories of noise to the datasets it generates:

#. **Row-based noise:** Errors in the inclusion or exclusion of entire rows of
   data, such as duplication or omission
#. **Column-based noise:** Errors in data entry for an individual field within a
   row, such as miswriting or incorrectly selecting responses

Each type of row-based noise operates on the entire dataset (selecting rows to
include or exclude), while each type of column-based noise operates on one
column of data at a time (selecting cells within that column to noise).
Currently, errors added in different columns are independent of each other.

Available Noise Types
---------------------

The table lists all the available noise types, but not every
type of noise will necessarily be applied to every dataset or every column. The
configuration determines which noise types are actually used. The "Noise Type"
column shows the name of the noise type in the configuration system.

.. list-table:: Types of row-based noise (``row_noise``)
  :widths: 1 2 5 
  :header-rows: 1

  * - Noise Type
    - Config key 
    - Example cause
  * - Omit a row
    - ``omit_row``
    - Neglecting to file a tax form on time
  * - Duplicate a row
    - ``duplicate_row``
    - College student being reported both at their university and their guardian's
      home address


.. list-table:: Types of column-based noise (``column_noise``)
  :widths: 1 2 5 
  :header-rows: 1

  * - Noise Type
    - Config key
    - Example cause
  * - Leave a field blank
    - ``leave_blank``
    - Forgetting to write your name on the designated line
  * - Choose the wrong option from a fixed set of options
    - ``choose_wrong_option``
    - Marking the "Male" box when you meant "Female"
  * - Optical character recognition (OCR) error
    - ``make_ocr_errors``
    - Misreading an 'S' instead of a '5'
  * - Phonetic error
    - ``make_phonetic_errors``
    - Mishearing a 't' for a 'd'
  * - Typographic errors
    - ``make_typos``
    - Accidentally typing an 'l' instead of a 'k' because they are 
      right next to each other on a QWERTY keyboard
  * - Nicknames
    - ``use_nickname``
    - Writing 'Alex' instead of legal name 'Alexander'
  * - Fake names
    - ``use_fake_name``
    - Using 'Mr' rather than actual first name
  * - Numeric miswriting 
    - ``write_wrong_digits``
    - Writing "2022" instead of "2002" in your date of birth
  * - Age miswriting
    - ``misreport_age``
    - Reporting that you are 28 years old when you are actually 27
  * - ZIP code miswriting
    - ``write_wrong_zipcode_digits``
    - Writing ZIP code 98118 when you actually live in 98112
  * - Copy from within household
    - ``copy_from_household_member``
    - Accidentally writing the age of another person in your household in the line for your age
  * - Month and day swap
    - ``swap_morth_and_day``
    - Reporting 17/05/1976 when a survey asks for the date in MM/DD/YYYY format


Default Noise Types for Each Column
-----------------------------------

Noise Function Details
----------------------

.. toctree::
  :maxdepth: 2

  row_noise
  column_noise
