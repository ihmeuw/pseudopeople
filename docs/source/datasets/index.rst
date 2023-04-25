.. _datasets_main:

========
Datasets
========

Here we cover the realistic simulated datasets, which are analogous to "real world" administrative records such as tax documents
and routinely generated files of social security numbers, that users can generate using Pseudopeople for developing and testing Entity
Resolution algorithms and software. 

Each of the datasets that can be generated using Pseudopeople have "noise" added to them, thereby realistically 
simulating how administrative records can be corrupted or distorted, which creates challenges in linking those 
records. To read more about the different kinds of noise that can be applied to the different datasets, please see the
`Noise page <https://pseudopeople.readthedocs.io/en/latest/noise_functions/index.html#noise-functions>`_.

The below table offers a list of the datasets that can be generated. Each row of a given dataset represents
an individual simulant, with the columns representing different simulant attributes, such as name, age, sex, et cetera.


.. contents::
   :depth: 2
   :local:
   :backlinks: none


US Decennial Census
-------------------
The Decennial Census dataset is a simulated enumeration of the US Census Bureau's Decennial Census of Population and Housing. The years
that have been simulated are 2020, 2030, and 2040. To find out more about the Decennial Census, please visit the Decennial Census
`homepage <https://www.census.gov/programs-surveys/decennial-census.html>`_.   

Generate Decennial Census data with :func:`pseudopeople.generate_decennial_census`

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes    
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise functions; intended use is "ground truth" for testing and validation. 
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
     - Formatted as YYYY-MM-DD.
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
     -    
   * - Physical address state
     - :code:`state`  
     - 
   * - Physical address ZIP code
     - :code:`zipcode`
     - 
   * - Relationship to reference person
     - :code:`relationship_to_reference_person` 
     - Possible values for this indicator include:
       Reference person; Biological child; Adopted child; Stepchild; Sibling; Parent; Grandchild; Parent-in-law; Child-in-law; Other relative;
       Roommate; Foster child; and Other nonrelative.
   * - Sex 
     - :code:`sex`  
     - Binary; "male" or "female".
   * - Race/ethnicity
     - :code:`race_ethnicity` 
     - The exhaustive and mutually exclusive categories for the single composite "race/ethnicity" indicator are as follows:
       White; Black; Latino; American Indian and Alaskan Native (AIAN); Asian; Native Hawaiian and Other Pacific Islander (NHOPI); and
       Multiracial or Some Other Race. 

American Community Survey (ACS)
-------------------------------
ACS is one of two household surveys that can currently be simulated using Pseudopeople. ACS is an ongoing household survey conducted by the US Census
Bureau that gathers information on a rolling basis about American community populations. Information collected includes ancestry, citizenship,
education, income, language proficienccy, migration, employment, disability, and housing characteristics. To find out more about ACS, please
visit the `ACS homepage <https://www.census.gov/programs-surveys/acs/about.html>`_.

Generate ACS data with :func:`pseudopeople.generate_american_community_survey`

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise functions; intended use is "ground truth" for testing and validation. 
   * - Household ID 
     - :code:`household_id` 
     - Not affected by noise functions; intended use is "ground truth" for testing and validation.
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
     - Formatted as YYYY-MM-DD.
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
     - 
   * - Physical address state
     - :code:`state`  
     - 
   * - Physical address ZIP code
     - :code:`zipcode`
     - 
   * - Sex 
     - :code:`sex`  
     - Binary; "male" or "female"
   * - Race/ethnicity
     - :code:`race_ethnicity` 
     - The exhaustive and mutually exclusive categories for the single composite "race/ethnicity" indicator are as follows:
       White; Black; Latino; American Indian and Alaskan Native (AIAN); Asian; Native Hawaiian and Other Pacific Islander (NHOPI); and
       Multiracial or Some Other Race.  

Current Population Survey (CPS)
-------------------------------
CPS is another household survey that can be simulated using Pseudopeople. CPS is conducted jointly by the US Census Bureau and the US 
Bureau of Labor Statistics. CPS collects labor force data, such as annual work activity and income, veteran status, school enrollment, 
contingent employment, worker displacement, job tenure, and more. To find out more about CPS, please visit the 
`CPS homepage <https://www.census.gov/programs-surveys/cps.html>`_. 

Generate CPS data with :func:`pseudopeople.generate_current_population_survey`

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise functions; intended use is "ground truth" for testing and validation. 
   * - Household ID 
     - :code:`household_id` 
     - Not affected by noise functions; intended use is "ground truth" for testing and validation.
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
     - Formatted as YYYY-MM-DD.
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
     - 
   * - Physical address state
     - :code:`state`  
     - 
   * - Physical address ZIP code
     - :code:`zipcode`
     - 
   * - Sex 
     - :code:`sex`  
     - Binary; "male" or "female"
   * - Race/ethnicity
     - :code:`race_ethnicity` 
     - The exhaustive and mutually exclusive categories for the single composite "race/ethnicity" indicator are as follows:
       White; Black; Latino; American Indian and Alaskan Native (AIAN); Asian; Native Hawaiian and Other Pacific Islander (NHOPI); and
       Multiracial or Some Other Race.  



Women, Infants, and Children (WIC)
----------------------------------
The Special Supplemental Nutrition Program for Women, Infants, and Children (WIC) is a government benefits program designed to support mothers and young
children. The main qualifications are income and the presence of young children in the home. To find out more about this service, please visit the `WIC 
homepage <https://www.fns.usda.gov/wic>`_.

Pseudopeople can generate a simulated version of the administrative data that would be recorded by WIC. This is a yearly file of information about all 
simulants enrolled in the program as of the end of that year.

Generate WIC data with :func:`pseudopeople.generate_women_infants_and_children` 

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise functions; intended use is "ground truth" for testing and validation. 
   * - Household ID 
     - :code:`household_id` 
     - Not affected by noise functions; intended use is "ground truth" for testing and validation.
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
     - Formatted as YYYY-MM-DD.
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
     - 
   * - Physical address state
     - :code:`state`  
     - 
   * - Physical address ZIP code
     - :code:`zipcode`
     - 
   * - Sex 
     - :code:`sex`  
     - Binary; "male" or "female"
   * - Race/ethnicity
     - :code:`race_ethnicity` 
     - The exhaustive and mutually exclusive categories for the single composite "race/ethnicity" indicator are as follows:
       White; Black; Latino; American Indian and Alaskan Native (AIAN); Asian; Native Hawaiian and Other Pacific Islander (NHOPI); and
       Multiracial or Some Other Race.  


Social Security Administration
------------------------------
The Social Security Administration (SSA) is the US federal government agency that administers Social Security, the social insurance program
that consists of retirement, disability and survivor benefits. To find out more about this program, visit the `SSA homepage <https://www.ssa.gov/about-ssa>`_.

Pseudopeople can generate a simulated version of a subset of the administrative data that would be recorded by SSA. Currently, the simulated
SSA data includes records of SSA creation and dates of death.

Generate SSA data with :func:`pseudopeople.generate_social_security` 

The following columns are included in this dataset:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise functions; intended use is "ground truth" for PRL tracking.  
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
     - Formatted as YYYY-MM-DD.
   * - Social security number
     - :code:`ssn`
     - By default, the SSN column in the SSA dataset has no :ref:`column-based noise <column_noise>`.
       However, it can be :ref:`configured <configuration_main>` to have noise if desired.
   * - Date of event
     - :code:`event_date`
     - Formatted as YYYY-MM-DD.  
   * - Type of event
     - :code:`event_type`
     - Possible values are "Creation" and "Death". 


Tax forms: W-2 & 1099
---------------------
Administrative data reported in annual tax forms, such as W-2s and 1099s, can also be simulated by Pseudopeople. 1099 forms are used for independent 
contractors or self-employed individuals, while a W-2 form is used for employees (whose employer withholds payroll taxes from their earnings).

Generate W-2 and 1099 data with :func:`pseudopeople.generate_taxes_w2_and_1099` 

The following columns are included in these datasets:

.. list-table:: **Dataset columns**
   :header-rows: 1

   * - Attribute Name
     - Column Name
     - Notes
   * - Unique simulant ID
     - :code:`simulant_id`
     - Not affected by noise functions; intended use is "ground truth" for testing and validation. 
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
     - Formatted as YYYY-MM-DD.
   * - Mailing address street number
     - :code:`mailing_address_street_number`
     - 
   * - Mailing address street name
     - :code:`mailing_address_street_name`
     - 
   * - Mailing address unit number
     - :code:`mailing_address_unit_number`
     - 
   * - Mailing address city
     - :code:`mailing_address_city`    
     - 
   * - Mailing address state
     - :code:`mailing_address_state`  
     - 
   * - Mailing address ZIP code
     - :code:`mailing_address_zipcode`
     - 
   * - Social security number 
     - :code:`ssn`
     - 
   * - Income 
     - :code:`income`
     - 
   * - Employer ID 
     - :code:`employer_id`
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
     - 
   * - Employer state
     - :code:`employer_state`  
     - 
   * - Employer ZIP code
     - :code:`employer_zipcode`
     - 
   * - Type of tax form 
     - :code:`tax_form`
     - Possible values are "W2" or "1099".

Tax form: 1040
--------------
As with data collected from W-2 and 1099 forms, Pseudopeople will also enable the simulation of administrative records from 1040 forms, which are
also reported to the IRS on an annual basis. This feature has not yet been implemented, so please stay tuned for more information! 
