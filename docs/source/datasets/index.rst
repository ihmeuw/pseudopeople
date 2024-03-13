.. _datasets_main:

========
Datasets
========

Here we cover the realistic simulated datasets, which are analogous to "real world" administrative records such as tax documents
and routinely generated files of social security numbers, that users can generate using Pseudopeople for developing and testing Entity
Resolution algorithms and software.

Each of the datasets that can be generated using pseudopeople has "noise" added to it, thereby realistically
simulating how population data can be corrupted or distorted, which creates challenges in linking those
records. To read more about the different kinds of noise that can be applied to the different datasets, please see the
:ref:`Noise page <noise_main>`.

pseudopeople generates datasets about a single simulated US population, which is followed through
time between January 1st, 2019 and May 1st, 2041.
Most datasets are yearly and can be generated for any year between 2019 and 2041 (inclusive),
though 2041 data will be partial.

There are two kinds of street addresses present in pseudopeople datasets:
physical addresses and mailing addresses.
A **physical address** represents the physical location where a simulant lives,
which is where they are recorded in the Decennial Census and surveys.
A **mailing address** represents the address a simulant uses to receive mail,
which may be different -- for example, a PO box.
Mailing addresses, not physical addresses, are recorded in tax filings.

Note that in the small-scale simulated population that is available by default, these addresses all have their 
city/state/zip code set to the fictitious location of Anytown, WA 00000. This is to ensure that linking is not 
unrealistically easy with the sample population (i.e., using these attributes to eliminate clear non-matches is
not possible, as they are all identical). To read more about obtaining large-scale data with more realistic city, state, and zip code data, please see 
:ref:`Simulated populations <simulated_populations_main>`.

Some fields are not applicable to every record in a simulated dataset,
so some columns may contain "missing" values, even if no noise has been added to the data.
For example, most addresses do not have a unit number, and some do not have a street number, so the 
:code:`unit_number` and/or :code:`street_number` fields will be "missing" for many rows in any dataset that contains addresses.
Similarly, columns pertaining to spouse or dependents in the 1040 tax dataset are not applicable to every simulant, so these columns also contain missing values.
Values that are missing because they are not applicable are represented by :code:`numpy.nan`.

The datasets that can be generated are listed below.

.. contents::
   :depth: 2
   :local:
   :backlinks: none


US Decennial Census
-------------------
The Decennial Census dataset is a simulated enumeration of the US Census Bureau's Decennial Census of Population and Housing.
To find out more about the Decennial Census, please visit the Decennial Census
`homepage <https://www.census.gov/programs-surveys/decennial-census.html>`_.

It is only possible to generate Decennial Census data for decennial years -- 2020, 2030, and 2040.

Generate Decennial Census data with :func:`pseudopeople.generate_decennial_census`.

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Unique household ID
     - :code:`household_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - First name
     - :code:`first_name`
     -
   * - Middle initial
     - :code:`middle_initial`
     -
   * - Last name
     - :code:`last_name`
     -
   * - Age
     - :code:`age`
     - Rounded down to an integer.
   * - Date of birth
     - :code:`date_of_birth`
     - Formatted as MM/DD/YYYY.
   * - Physical address street number
     - :code:`street_number`
     -
   * - Physical address street name
     - :code:`street_name`
     -
   * - Physical address unit number
     - :code:`unit_number`
     -
   * - Physical address city
     - :code:`city`
     - Default simulated population always has value "Anytown"
   * - Physical address state
     - :code:`state`
     - Default simulated population always has value "WA"
   * - Physical address ZIP code
     - :code:`zipcode`
     - Default simulated population always has value "00000"
   * - Housing type
     - :code:`housing_type`
     - Possible values for housing type are "Household" for an individual
       household, or one of six different types of group quarters. The types of
       institutional group quarters are "Carceral", "Nursing home", and "Other
       institutional". The types of noninstitutional group quarters are
       "College", "Military", and "Other noninstitutional".
   * - Relationship to reference person
     - :code:`relationship_to_reference_person`
     - Possible values for this field include:
       "Reference person"; "Opposite-sex spouse"; "Opposite-sex unmarried
       partner"; "Same-sex spouse"; "Same-sex unmarried partner"; "Biological
       child"; "Adopted child"; "Stepchild"; "Sibling"; "Parent"; "Grandchild";
       "Parent-in-law"; "Child-in-law"; "Other relative"; "Roommate or
       housemate"; "Foster child"; "Other nonrelative"; "Institutionalized group
       quarters population"; and "Noninstitutionalized group quarters
       population".
   * - Sex
     - :code:`sex`
     - Binary; "male" or "female".
   * - Race/ethnicity
     - :code:`race_ethnicity`
     - The categories for the single composite "race/ethnicity" field are as follows:
       "White"; "Black"; "Latino"; "American Indian and Alaskan Native (AIAN)"; "Asian"; "Native Hawaiian and Other Pacific Islander (NHOPI)"; and
       "Multiracial or Some Other Race".
   * - Year
     - :code:`year`
     - Year in which data were collected; metadata that would not be collected directly; not affected by noise.

American Community Survey (ACS)
-------------------------------
ACS is one of two household surveys that can currently be simulated using Pseudopeople. ACS is an ongoing household survey conducted by the US Census
Bureau that gathers information on a rolling basis about American community populations. Information collected includes ancestry, citizenship,
education, income, language proficienccy, migration, employment, disability, and housing characteristics. To find out more about ACS, please
visit the `ACS homepage <https://www.census.gov/programs-surveys/acs/about.html>`_.

pseudopeople can generate ACS data for a user-specified year,
which will include records from simulated surveys conducted
throughout that calendar year.

Generate ACS data with :func:`pseudopeople.generate_american_community_survey`.

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Unique household ID
     - :code:`household_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Survey date
     - :code:`survey_date`
     - Date on which the survey was conducted; metadata that would not be collected directly; not affected by noise.
       Stored as a ``pandas.Timestamp``, which displays in YYYY-MM-DD format by default.
   * - First name
     - :code:`first_name`
     -
   * - Middle initial
     - :code:`middle_initial`
     -
   * - Last name
     - :code:`last_name`
     -
   * - Age
     - :code:`age`
     - Rounded down to an integer.
   * - Date of birth
     - :code:`date_of_birth`
     - Formatted as MM/DD/YYYY.
   * - Physical address street number
     - :code:`street_number`
     -
   * - Physical address street name
     - :code:`street_name`
     -
   * - Physical address unit number
     - :code:`unit_number`
     -
   * - Physical address city
     - :code:`city`
     - Default simulated population always has value "Anytown"
   * - Physical address state
     - :code:`state`
     - Default simulated population always has value "WA"
   * - Physical address ZIP code
     - :code:`zipcode`
     - Default simulated population always has value "00000"
   * - Housing type
     - :code:`housing_type`
     - Possible values for housing type are "Household" for an individual
       household, or one of six different types of group quarters. The types of
       institutional group quarters are "Carceral", "Nursing home", and "Other
       institutional". The types of noninstitutional group quarters are
       "College", "Military", and "Other noninstitutional".
   * - Relationship to reference person
     - :code:`relationship_to_reference_person`
     - Possible values for this field include:
       "Reference person"; "Opposite-sex spouse"; "Opposite-sex unmarried
       partner"; "Same-sex spouse"; "Same-sex unmarried partner"; "Biological
       child"; "Adopted child"; "Stepchild"; "Sibling"; "Parent"; "Grandchild";
       "Parent-in-law"; "Child-in-law"; "Other relative"; "Roommate or
       housemate"; "Foster child"; "Other nonrelative"; "Institutionalized group
       quarters population"; and "Noninstitutionalized group quarters
       population".
   * - Sex
     - :code:`sex`
     - Binary; "male" or "female"
   * - Race/ethnicity
     - :code:`race_ethnicity`
     - The categories for the single composite "race/ethnicity" field are as follows:
       "White"; "Black"; "Latino"; "American Indian and Alaskan Native (AIAN)"; "Asian"; "Native Hawaiian and Other Pacific Islander (NHOPI)"; and
       "Multiracial or Some Other Race".

Current Population Survey (CPS)
-------------------------------
CPS is another household survey that can be simulated using Pseudopeople. CPS is conducted jointly by the US Census Bureau and the US
Bureau of Labor Statistics. CPS collects labor force data, such as annual work activity and income, veteran status, school enrollment,
contingent employment, worker displacement, job tenure, and more. To find out more about CPS, please visit the
`CPS homepage <https://www.census.gov/programs-surveys/cps.html>`_.

pseudopeople can generate CPS data for a user-specified year,
which will include records from simulated surveys conducted
throughout that calendar year.

Generate CPS data with :func:`pseudopeople.generate_current_population_survey`.

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Unique household ID
     - :code:`household_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Survey date
     - :code:`survey_date`
     - Date on which the survey was conducted; metadata that would not be collected directly; not affected by noise.
       Stored as a ``pandas.Timestamp``, which displays in YYYY-MM-DD format by default.
   * - First name
     - :code:`first_name`
     -
   * - Middle initial
     - :code:`middle_initial`
     -
   * - Last name
     - :code:`last_name`
     -
   * - Age
     - :code:`age`
     - Rounded down to an integer.
   * - Date of birth
     - :code:`date_of_birth`
     - Formatted as MM/DD/YYYY.
   * - Physical address street number
     - :code:`street_number`
     -
   * - Physical address street name
     - :code:`street_name`
     -
   * - Physical address unit number
     - :code:`unit_number`
     -
   * - Physical address city
     - :code:`city`
     - Default simulated population always has value "Anytown"
   * - Physical address state
     - :code:`state`
     - Default simulated population always has value "WA"
   * - Physical address ZIP code
     - :code:`zipcode`
     - Default simulated population always has value "00000"
   * - Sex
     - :code:`sex`
     - Binary; "male" or "female"
   * - Race/ethnicity
     - :code:`race_ethnicity`
     - The categories for the single composite "race/ethnicity" field are as follows:
       "White"; "Black"; "Latino"; "American Indian and Alaskan Native (AIAN)"; "Asian"; "Native Hawaiian and Other Pacific Islander (NHOPI)"; and
       "Multiracial or Some Other Race".



Women, Infants, and Children (WIC)
----------------------------------
The Special Supplemental Nutrition Program for Women, Infants, and Children (WIC) is a government benefits program designed to support mothers and young
children. The main qualifications are income and the presence of young children in the home. To find out more about this service, please visit the `WIC
homepage <https://www.fns.usda.gov/wic>`_.

pseudopeople can generate a simulated version of the administrative data that would be recorded by WIC. This is a yearly file of information about all
simulants enrolled in the program as of the end of that year.
For the final year available, 2041, the file includes those enrolled as of May 1st, because this is the end of our simulated timespan.

Generate WIC data with :func:`pseudopeople.generate_women_infants_and_children`.

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Unique household ID
     - :code:`household_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - First name
     - :code:`first_name`
     -
   * - Middle initial
     - :code:`middle_initial`
     -
   * - Last name
     - :code:`last_name`
     -
   * - Date of birth
     - :code:`date_of_birth`
     - Formatted as MMDDYYYY.
   * - Physical address street number
     - :code:`street_number`
     -
   * - Physical address street name
     - :code:`street_name`
     -
   * - Physical address unit number
     - :code:`unit_number`
     -
   * - Physical address city
     - :code:`city`
     - Default simulated population always has value "Anytown"
   * - Physical address state
     - :code:`state`
     - Default simulated population always has value "WA"
   * - Physical address ZIP code
     - :code:`zipcode`
     - Default simulated population always has value "00000"
   * - Sex
     - :code:`sex`
     - Binary; "male" or "female"
   * - Race/ethnicity
     - :code:`race_ethnicity`
     - The categories for the single composite "race/ethnicity" field are as follows:
       "White"; "Black"; "Latino"; "American Indian and Alaskan Native (AIAN)"; "Asian"; "Native Hawaiian and Other Pacific Islander (NHOPI)"; and
       "Multiracial or Some Other Race".
   * - Year
     - :code:`year`
     - Year in which benefits were received; metadata that would not be collected directly; not affected by noise.


Social Security Administration
------------------------------
The Social Security Administration (SSA) is the US federal government agency that administers Social Security, the social insurance program
that consists of retirement, disability and survivor benefits. To find out more about this program, visit the `SSA homepage <https://www.ssa.gov/about-ssa>`_.

pseudopeople can generate a simulated version of a subset of the administrative data that would be recorded by SSA.
Currently, the simulated SSA data includes records of SSN creation and dates of death.
This is a yearly data file that is **cumulative** -- when you specify a year, you will recieve all records *up to the end of*
that year.

The simulated SSA data files will not include records about simulants who died before 2019 (the start of our simulated timespan).
Therefore, while SSA data files can be generated for years prior to 2019, they will only include records for SSN creation,
and only for simulants who were still alive in 2019.

Generate SSA data with :func:`pseudopeople.generate_social_security`.

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Social security number
     - :code:`ssn`
     - By default, the SSN column in the SSA dataset has no :ref:`column-based noise <column_noise>`.
       However, it can be :ref:`configured <configuration_main>` to have noise if desired.
   * - First name
     - :code:`first_name`
     -
   * - Middle name
     - :code:`middle_name`
     -
   * - Last name
     - :code:`last_name`
     -
   * - Date of birth
     - :code:`date_of_birth`
     - Formatted as YYYYMMDD.
   * - Sex
     - :code:`sex`
     - Binary; "male" or "female"
   * - Type of event
     - :code:`event_type`
     - Possible values are "Creation" and "Death".
   * - Date of event
     - :code:`event_date`
     - Formatted as YYYYMMDD.


Tax forms: W-2 & 1099
---------------------
Administrative data reported in annual tax forms, such as W-2s and 1099s, can also be simulated by Pseudopeople. 1099 forms are used for independent
contractors or self-employed individuals, while a W-2 form is submitted by an employer for their employee (as the employer withholds payroll taxes from employee earnings).

pseudopeople can generate a simulated version of the data collected by W-2 and 1099 forms.
This is a yearly dataset, where the user-specified year is the **tax year** of the data.
That is, the data for 2022 will be the result of tax forms filed in early 2023.
Tax data can be generated for tax years 2019 through 2040 (inclusive).

Generate W-2 and 1099 data with :func:`pseudopeople.generate_taxes_w2_and_1099`.

The following columns are included in these datasets:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Unique household ID
     - :code:`household_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Employer ID
     - :code:`employer_id`
     -
   * - Social security number
     - :code:`ssn`
     -
   * - Wages
     - :code:`wages`
     -
   * - Employer Name
     - :code:`employer_name`
     -
   * - Employer street number
     - :code:`employer_street_number`
     -
   * - Employer street name
     - :code:`employer_street_name`
     -
   * - Employer unit number
     - :code:`employer_unit_number`
     -
   * - Employer city
     - :code:`employer_city`
     - Default simulated population always has value "Anytown"
   * - Employer state
     - :code:`employer_state`
     - Default simulated population always has value "WA"
   * - Employer ZIP code
     - :code:`employer_zipcode`
     - Default simulated population always has value "00000"
   * - First name
     - :code:`first_name`
     -
   * - Middle initial
     - :code:`middle_initial`
     -
   * - Last name
     - :code:`last_name`
     -
   * - Mailing address street number
     - :code:`mailing_address_street_number`
     -
   * - Mailing address street name
     - :code:`mailing_address_street_name`
     -
   * - Mailing address unit number
     - :code:`mailing_address_unit_number`
     -
   * - Mailing address PO Box
     - :code:`mailing_address_po_box`
     -
   * - Mailing address city
     - :code:`mailing_address_city`
     - Default simulated population always has value "Anytown"
   * - Mailing address state
     - :code:`mailing_address_state`
     - Default simulated population always has value "WA"
   * - Mailing address ZIP code
     - :code:`mailing_address_zipcode`
     - Default simulated population always has value "00000"
   * - Type of tax form
     - :code:`tax_form`
     - Possible values are "W2" or "1099".
   * - Tax year
     - :code:`tax_year`
     - Year for which tax data were collected; metadata that would not be collected directly; not affected by noise.


Tax form: 1040
--------------
As with data collected from W-2 and 1099 forms, pseudopeople enables the simulation of administrative records from 1040 forms, which are
also reported to the IRS on an annual basis. While W-2 forms are submitted by an employer to the IRS, 1040 forms are submitted by the 
employee. To find out more about the 1040 tax form, visit the `IRS information page <https://www.irs.gov/instructions/i1040gi>`_.

A single row in a pseudopeople-generated 1040 dataset may contain information about several
simulants: the primary filer, the primary filer's joint filer (spouse) if they are married filing
jointly, and up to four claimed dependents.
When not applicable, all relevant fields are :code:`numpy.nan`;
for example, a row representing a 1040 filed by only one simulant, without a joint filer,
would have missingness in all joint filer columns.

If a simulant claims fewer than four dependents, they will be filled in starting
with :code:`dependent_1`.
For example, a simulant claiming three dependents would have missingness in all
:code:`dependent_4` columns.
A simulant may claim more than four dependents, but only four will appear in the
dataset; the rest are omitted.

All columns not otherwise labeled are about the primary filer;
for example, the :code:`first_name` column is the first name of the primary filer.
The :code:`simulant_id` and :code:`household_id` columns represent the "ground truth"
of which simulant is the primary filer, and which household *that primary filer* lives
in.
It is not guaranteed that all simulants described in a 1040 row live in the same household;
for example, college students may be claimed as dependents while living elsewhere.

A single simulant can appear in multiple rows in this dataset,
for example if they filed a 1040 and were also claimed as a dependent on another
simulant's 1040.

This is a yearly dataset, where the user-specified year is the **tax year** of the data.
1040 data can be generated for tax years 2019 through 2040 (inclusive).

Generate 1040 data with :func:`pseudopeople.generate_taxes_1040`.

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - Unique household ID
     - :code:`household_id`
     - Not affected by noise; intended use is "ground truth" for testing and validation; consistent across all
       datasets.
   * - First name
     - :code:`first_name`
     -
   * - Middle initial
     - :code:`middle_initial`
     -
   * - Last name
     - :code:`last_name`
     -
   * - Social Security Number (SSN)
     - :code:`ssn`
     - Individual Taxpayer Identification Number (ITIN) if no SSN
   * - Mailing address street number
     - :code:`mailing_address_street_number`
     -
   * - Mailing address street name
     - :code:`mailing_address_street_name`
     -
   * - Mailing address unit number
     - :code:`mailing_address_unit_number`
     -
   * - Mailing address PO box
     - :code:`mailing_address_po_box`
     -
   * - Mailing address city
     - :code:`mailing_address_city`
     - Default simulated population always has value "Anytown"
   * - Mailing address state
     - :code:`mailing_address_state`
     - Default simulated population always has value "WA"
   * - Mailing address ZIP code
     - :code:`mailing_address_zipcode`
     - Default simulated population always has value "00000"
   * - Joint filer first name
     - :code:`spouse_first_name`
     -
   * - Joint filer middle initial
     - :code:`spouse_middle_initial`
     -
   * - Joint filer last name
     - :code:`spouse_last_name`
     -
   * - Joint filer social security number
     - :code:`spouse_ssn`
     - Individual Taxpayer Identification Number (ITIN) if no SSN
   * - Dependent 1 first name
     - :code:`dependent_1_first_name`
     -
   * - Dependent 1 last name
     - :code:`dependent_1_last_name`
     -
   * - Dependent 1 Social Security Number (SSN)
     - :code:`dependent_1_ssn`
     - Individual Taxpayer Identification Number (ITIN) if no SSN
   * - Dependent 2 first name
     - :code:`dependent_2_first_name`
     -
   * - Dependent 2 last name
     - :code:`dependent_2_last_name`
     -
   * - Dependent 2 social security number
     - :code:`dependent_2_ssn`
     - Individual Taxpayer Identification Number (ITIN) if no SSN
   * - Dependent 3 first name
     - :code:`dependent_3_first_name`
     -
   * - Dependent 3 last name
     - :code:`dependent_3_last_name`
     -
   * - Dependent 3 social security number
     - :code:`dependent_3_ssn`
     - Individual Taxpayer Identification Number (ITIN) if no SSN
   * - Dependent 4 first name
     - :code:`dependent_4_first_name`
     -
   * - Dependent 4 last name
     - :code:`dependent_4_last_name`
     -
   * - Dependent 4 social security number
     - :code:`dependent_4_ssn`
     - Individual Taxpayer Identification Number (ITIN) if no SSN
   * - Tax year
     - :code:`tax_year`
     - Year for which tax data were collected; metadata that would not be collected directly; not affected by noise.
