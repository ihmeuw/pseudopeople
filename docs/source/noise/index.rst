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
  * - Duplicate with guardian
    - ``duplicate_with_guardian``
    - Filling out a survey questionnaire for your child that lives in college housing, while they simultaneously
      fill out the same questionnaire for themselves
  * - Do not respond
    - ``do_not_respond``
    - Not returning the American Community Survey questionnaire that the US Census Bureau sent you
  * - Omit a row
    - ``omit_row``
    - Losing data because of an administrative error

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
  * - Copy from household member
    - ``copy_from_household_member``
    - Accidentally writing your daughter's age in a box that asked about your son's age on a survey questionnaire
  * - Use a nickname
    - ``use_nickname``
    - Writing 'Alex' instead of legal name 'Alexander'
  * - Use a fake name
    - ``use_fake_name``
    - Using "Mr" rather than actual first name
  * - Swap month and day
    - ``swap_month_and_day``
    - Reporting 17/05/1976 when a survey asks for the date in MM/DD/YYYY format
  * - Misreport age
    - ``misreport_age``
    - Reporting that you are 28 years old when you are actually 27
  * - Write the wrong digits
    - ``write_wrong_digits``
    - Writing "732 Main St" as your street address instead of "932 Main St"
  * - Write the wrong ZIP code digits
    - ``write_wrong_zipcode_digits``
    - Writing ZIP code 98118 when you actually live in 98112
  * - Make phonetic errors
    - ``make_phonetic_errors``
    - Mishearing a 't' for a 'd'
  * - Make Optical Character Recognition (OCR) errors
    - ``make_ocr_errors``
    - Misreading an 'S' instead of a '5'
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
    - Applicable datasets
    - Types of noise
    - Notes
  * - First name
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040, SSA
    - Leave a field blank, use a nickname, use a fake name, make phonetic errors, make OCR errors, make typos
    - In the 1040 form, the same noise types apply to the first name columns for the joint filer and dependents
  * - Middle name
    - SSA
    - Leave a field blank, use a nickname, use a fake name, make phonetic errors, make OCR errors, make typos
    - Middle names use the same lists of nicknames and fake names used for first names
  * - Middle initial
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, make phonetic errors, make OCR errors, make typos
    - In the 1040 form, the same noise types apply to the middle initial columns for the joint filer and dependents
  * - Last name
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040, SSA
    - Leave a field blank, use a fake name, make phonetic errors, make OCR errors, make typos
    - Last names use a different list of fake names than the list for first names. In the 1040 form, the same noise types apply to the last name columns for the joint filer and dependents
  * - Age
    - Decennial Census, ACS, CPS
    - Leave a field blank, copy from household member, misreport age, make OCR errors, make typos
    -
  * - Date of birth
    - Decennial Census, ACS, CPS, WIC, SSA
    - Leave a field blank, copy from household member, swap month and day, write the wrong digits, make OCR errors, make typos
    -
  * - Street number for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, write the wrong digits, make OCR errors, make typos
    - Noise for all types of addresses works in the same way
  * - Street name for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, make phonetic errors, make OCR errors, make typos
    - Noise for all types of addresses works in the same way
  * - Unit number for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, write the wrong digits, make OCR errors, make typos
    - Noise for all types of addresses works in the same way
  * - PO Box for mailing address
    - W-2 and 1099, 1040
    - Leave a field blank, write the wrong digits, make OCR errors, make typos
    -
  * - City name for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, make phonetic errors, make OCR errors, make typos
    - Noise for all types of addresses works in the same way
  * - State for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, choose the wrong option
    - Noise for all types of addresses works in the same way
  * - ZIP code for any address (physical, mailing, or employer)
    - Decennial Census, ACS, CPS, WIC, W-2 and 1099, 1040
    - Leave a field blank, write the wrong zipcode digits, make OCR errors, make typos
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
    - Borrow a social security number, leave a field blank, copy from household member, write the wrong digits, make OCR errors, make typos
    - Note that "borrow a social security number" only applies to the W-2 and 1099 dataset, and by default, "copy from household member" noise is turned off in this dataset (but can be turned on if desired).
      In the SSA dataset, the SSN column has no column-based noise by default (but can be configured to have noise if desired).
      In the 1040 form, the same noise types apply to the SSN columns for the joint filer and dependents.
  * - Wages
    - W-2 and 1099
    - Leave a field blank, write the wrong digits, make OCR errors, make typos
    -
  * - Employer ID
    - W-2 and 1099
    - Leave a field blank, write the wrong digits, make OCR errors, make typos
    -
  * - Employer name
    - W-2 and 1099
    - Leave a field blank, make OCR errors, make typos
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
    - Leave a field blank, swap month and day, write the wrong digits, make OCR errors, make typos
    -

.. _noise_type_details:

Noise type details
----------------------

For more details on the different row-based and column-based noise types covered above, please follow the links
below. 

.. toctree::
  :maxdepth: 2

  row_noise
  column_noise
