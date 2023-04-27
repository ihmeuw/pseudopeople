.. _noise_main:

=====
Noise
=====

In order to have a realistic challenge with entity resolution, it is essential
to add noise to the simulated data. "Noise" refers to various types of errors
introduced into the data and may also be called "corruption" or "distortion." By
default, pseudopeople applies noise to the simulated datasets using some
reasonable settings. If desired, the user can change the noise settings through
the configuration system---see the :ref:`Configuration section <configuration_main>` 
for details.

.. contents::
   :depth: 2
   :local:
   :backlinks: entry


.. _categories_of_noise:

Categories of noise
-------------------

pseudopeople can add two broad categories of noise to the datasets it generates:

#. **Row-based noise:** Errors in the inclusion or exclusion of entire rows of
   data, such as duplication or omission
#. **Column-based noise:** Errors in data entry for an individual field within a
   row, such as miswriting or incorrectly selecting responses

Each type of row-based noise operates on the entire dataset (selecting rows to
include or exclude), while each type of column-based noise operates on one
column of data at a time (selecting cells within that column to noise).
Currently, errors added in different columns are independent of each other.

.. _available_noise_types:

Available noise types
---------------------

These tables list all the available noise types, but not every
type of noise will necessarily be applied to every dataset or every column.
Noise types are applied in the order they are listed here.
The "Config key" column shows the name of the noise type in the :ref:`configuration system <configuration_main>`.

.. list-table:: Types of row-based noise (``row_noise``)
  :widths: 1 2 5 
  :header-rows: 1

  * - Noise type
    - Config key 
    - Example cause
  * - Omit a row
    - ``omit_row``
    - Neglecting to file a tax form on time

.. list-table:: Types of column-based noise (``column_noise``)
  :widths: 1 2 5 
  :header-rows: 1

  * - Noise type
    - Config key
    - Example cause
  * - "Borrowed" SSN
    - Not configurable
    - Using your housemate's SSN on a W-2 because you do not have one
  * - Leave a field blank
    - ``leave_blank``
    - Forgetting to write your name on the designated line
  * - Choose the wrong option
    - ``choose_wrong_option``
    - Marking the "Male" box when you meant "Female"
  * - Use a fake name
    - ``use_fake_name``
    - Using "Mr" rather than actual first name
  * - Misreport age
    - ``misreport_age``
    - Reporting that you are 28 years old when you are actually 27
  * - Write the wrong digits
    - ``write_wrong_digits``
    - Writing "732 Main St" as your street address instead of "932 Main St"
  * - Write the wrong ZIP code digits 
    - ``write_wrong_zipcode_digits``
    - Writing ZIP code 98118 when you actually live in 98112
  * - Make typos
    - ``make_typos``
    - Accidentally typing an "l" instead of a "k" because they are 
      right next to each other on a QWERTY keyboard


Noise types for each column
-----------------------------------

.. list-table:: Types of noise for each column
  :widths: 20 20 20 20
  :header-rows: 1

  * - Column name
    - Datasets present
    - Types of noise
    - Notes
  * - First name
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, SSA
    - Leave a field blank, use a fake name, make typos
    -
  * - Middle initial
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, SSA
    - Leave a field blank, make typos
    -
  * - Last name
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, SSA
    - Leave a field blank, use a fake name, make typos
    -
  * - Age
    - Decennial Census, ACS, CPS, W-2 and 1099
    - Leave a field blank, misreport age, make typos
    -
  * - Date of birth
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, SSA
    - Leave a field blank, write the wrong digits, make typos
    -
  * - Street number for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099
    - Leave a field blank, write the wrong digits, make typos
    - Noise for all types of addresses works in the same way
  * - Street name for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099
    - Leave a field blank, make typos
    - Noise for all types of addresses works in the same way
  * - Unit number for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099
    - Leave a field blank, write the wrong digits, make typos
    - Noise for all types of addresses works in the same way
  * - PO Box for mailing address
    - W-2 and 1099
    - Leave a field blank, write the wrong digits, make typos
    -
  * - City name for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099
    - Leave a field blank, make typos
    - Noise for all types of addresses works in the same way
  * - State for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099
    - Leave a field blank 
    - Noise for all types of addresses works in the same way
  * - ZIP code for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099
    - Leave a field blank, write the wrong zipcode digits, make typos
    -
  * - Relationship to head of household
    - Decennial Census
    - Leave a field blank
    -
  * - Sex
    - Decennial Census, ACS, CPS, WIC
    - Leave a field blank
    -
  * - Race/ethnicity
    - Decennial Census, ACS, CPS, WIC
    - Leave a field blank
    -
  * - SSN
    - W-2 and 1099, SSA
    - "Borrowed" SSN, leave a field blank, write the wrong digits, make typos
    - Note that "borrowed" SSN only applies to the W-2 and 1099 dataset
  * - Income / wages
    - W-2 and 1099
    - Leave a field blank, make typos
    - Note that wages and income are on separate tax forms and noise is applied to each separately
  * - Employer ID
    - W-2 and 1099
    - Leave a field blank, write the wrong digits, make typos
    - 
  * - Employer name
    - W-2 and 1099
    - Leave a field blank, make typos
    -
  * - Type of tax form
    - W-2 and 1099
    - Leave a field blank
    -
  * - Type of SSA event
    - SSA
    - Leave a field blank
    -
  * - Date of SSA event
    - SSA
    - Leave a field blank, write the wrong digits, make typos
    -


.. _noise_type_details:

Noise type details
----------------------

.. toctree::
  :maxdepth: 2

  row_noise
  column_noise
