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
  * - Borrow a social security number
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
  * - Use a nickname
    - ``use_nickname``
    - Writing 'Alex' instead of legal name 'Alexander'
  * - Misreport age
    - ``misreport_age``
    - Reporting that you are 28 years old when you are actually 27
  * - Write the wrong digits
    - ``write_wrong_digits``
    - Writing "732 Main St" as your street address instead of "932 Main St"
  * - Write the wrong ZIP code digits
    - ``write_wrong_zipcode_digits``
    - Writing ZIP code 98118 when you actually live in 98112
  * - Swap month and day
    - ``swap_month_and_day``
    - Reporting 17/05/1976 when a survey asks for the date in MM/DD/YYYY format
  * - Make typos
    - ``make_typos``
    - Accidentally typing an "l" instead of a "k" because they are
      right next to each other on a QWERTY keyboard
  * - Make Optical Character Recognition (OCR) errors
    - ``make_ocr_errors``
    - Misreading an 'S' instead of a '5'
  * - Make phonetic errors
    - ``make_phonetic_errors``
    - Mishearing a 't' for a 'd'


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
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040, SSA
    - Leave a field blank, use a fake name, use a nickname, make typos, make OCR errors, make phonetic errors
    - In the 1040 form, the same noise types apply to the first name columns for the joint filer and dependents
  * - Middle name
    - SSA
    - Leave a field blank, use a fake name, use a nickname, make typos, make OCR errors, make phonetic errors
    - 
  * - Middle initial
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, make typos, make OCR errors, make phonetic errors
    - In the 1040 form, the same noise types apply to the middle initial columns for the joint filer and dependents
  * - Last name
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040, SSA
    - Leave a field blank, use a fake name, make typos, make OCR errors, make phonetic errors
    - In the 1040 form, the same noise types apply to the last name columns for the joint filer and dependents
  * - Age
    - Decennial Census, ACS, CPS, W-2 and 1099
    - Leave a field blank, misreport age, make typos, make OCR errors
    -
  * - Date of birth
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, SSA
    - Leave a field blank, write the wrong digits, swap month and day, make typos, make OCR errors
    -
  * - Street number for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, write the wrong digits, make typos, make OCR errors
    - Noise for all types of addresses works in the same way
  * - Street name for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, make typos, make OCR errors, make phonetic errors
    - Noise for all types of addresses works in the same way
  * - Unit number for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, write the wrong digits, make typos, make OCR errors
    - Noise for all types of addresses works in the same way
  * - PO Box for mailing address
    - W-2 and 1099, 1040
    - Leave a field blank, write the wrong digits, make typos, make OCR errors
    -
  * - City name for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, make typos, make OCR errors, make phonetic errors
    - Noise for all types of addresses works in the same way
  * - State for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, choose the wrong option
    - Noise for all types of addresses works in the same way
  * - ZIP code for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, write the wrong zipcode digits, make typos, make OCR errors
    -
  * - Housing type
    - Decennial Census, ACS
    - Leave a field blank, choose the wrong option
    -
  * - Relationship to reference person
    - Decennial Census, ACS
    - Leave a field blank, choose the wrong option
    -
  * - Sex
    - Decennial Census, ACS, CPS, WIC
    - Leave a field blank, choose the wrong option
    -
  * - Race/ethnicity
    - Decennial Census, ACS, CPS, WIC
    - Leave a field blank, choose the wrong option
    -
  * - SSN
    - W-2 and 1099, 1040, SSA
    - Borrow a social security number, leave a field blank, write the wrong digits, make typos, make OCR errors
    - Note that 'borrow a social security number' only applies to the W-2 and 1099 dataset.
      In the 1040 form, the same noise types apply to the SSN columns for the joint filer and dependents
  * - Wages 
    - W-2 and 1099
    - Leave a field blank, write the wrong digits, make typos, make OCR errors
    - 
  * - Employer ID
    - W-2 and 1099
    - Leave a field blank, write the wrong digits, make typos, make OCR errors
    -
  * - Employer name
    - W-2 and 1099
    - Leave a field blank, make typos, make OCR errors
    -
  * - Type of tax form
    - W-2 and 1099
    - Leave a field blank, choose the wrong option
    -
  * - Type of SSA event
    - SSA
    - Leave a field blank, choose the wrong option
    -
  * - Date of SSA event
    - SSA
    - Leave a field blank, write the wrong digits, swap month and day, make typos, make OCR errors
    -


.. _noise_type_details:

Noise type details
----------------------

.. toctree::
  :maxdepth: 2

  row_noise
  column_noise
