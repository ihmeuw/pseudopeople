.. _datasets_main:

========
Datasets
========

Here we cover the realistic simulated datasets, which are analogous to "real world" administrative records such as tax documents
and routinely generated files of social security numbers, that users can generate using Pseudopeople for developing and testing Entity Resolution algorithms 
and software. 

Each of the datasets that can be generated using Pseudopeople have "noise" added to them, thereby realistically 
simulating how administrative records can be corrupted or distorted, which creates challenges in linking those 
records. To read more about the different kinds of noise that can be applied to the different datasets, please see the `Noise page <https://pseudopeople.readthedocs.io/en/latest/noise_functions/index.html#noise-functions>`_.

The below table offers a list of the datasets that can be generated. Each row of a given dataset represents
an individual simulant, with the columns representing different simulant attributes, such as name, age, sex, et cetera.


.. contents::
   :depth: 2
   :local:
   :backlinks: none


.. list-table:: **Available Datasets**
   :header-rows: 1
   :widths: 20

   * - Name
   * - | US Decennial Census
   * - | American Community Survey (ACS)
   * - | Current Population Survey (CPS)
   * - | Women, Infants, and Children (WIC) Administrative Data
   * - | Social Security Administration (SSA) Data
   * - | Tax W2 and 1099 Forms
   * - | Tax 1040 Form


US Decennial Census
-------------------

The Decennial Census dataset is a simulated enumeration of the US Census Bureau's Decennial Census of Population and Housing. The years
that have been simulated are 2020, 2030, and 2040. To find out more about the Decennial Census, please visit the Decennial Census
`homepage <https://www.census.gov/programs-surveys/decennial-census.html>`_.   

To find out more about how to generate a simulation of the Decennial Census using Pseudopeople, see :func:`pseudopeople.interface.generate_decennial_census`.

The following simulant attributes are included in this dataset:

.. list-table:: **Simulant attributes**
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

Household Surveys: ACS and CPS
------------------------------
There are two simulated household survey datasets that can be used: the American
Community Survey (ACS) and the Current Population Survey (CPS). 

ACS is an ongoing household survey conducted by the US Census Bureau that gathers information on a rolling basis about
American community populations. Information collected includes ancestry, citizenship, education, income, language proficienccy, migration, 
employment, disability, and housing characteristics. To find out more about ACS, please visit the `ACS homepage <https://www.census.gov/programs-surveys/acs/about.html>`_.

CPS is another household survey conducted by the US Census Bureau and the US Bureau of Labor Statistics. This survey is administered by Census 
Bureau field representatives across the country through both personal and telephone interviews. CPS collects labor force data, such as annual
work activity and income, veteran status, school enrollment, contingent employment, worker displacement, job tenure, and more. To find out more
about CPS, please visit the `CPS homepage <https://www.census.gov/programs-surveys/cps.html>`_. 

The following simulant attributes are included in these datasets:

.. list-table:: **Simulant attributes**
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
     - The following exhaustive and mutually exclusive categories for the single composite "race/ethnicity" indicator are as follows:
       White; Black; Latino; American Indian and Alaskan Native (AIAN); Asian; Native Hawaiian and Other Pacific Islander (NHOPI); and
       Multiracial or Some Other Race.  


WIC
---
The Special Supplemental Nutrition Program for Women, Infants, and Children (WIC) is a government benefits program designed to support mothers and young
children. The main qualifications are income and the presence of young children in the home. To find out more about this service, please visit the `WIC 
homepage <https://www.fns.usda.gov/wic>`_.

Pseudopeople can generate a simulated version of the administrative data that would be recorded by WIC. This is a yearly file of information about all 
simulants enrolled in the program as of the end of that year.

The following simulant attributes are included in this dataset:

.. list-table:: **Simulant attributes**
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
     - The following exhaustive and mutually exclusive categories for the single composite "race/ethnicity" indicator are as follows:
       White; Black; Latino; American Indian and Alaskan Native (AIAN); Asian; Native Hawaiian and Other Pacific Islander (NHOPI); and
       Multiracial or Some Other Race.  


Social Security
---------------
The Social Security Administration (SSA) is the US federal government agency that administers Social Security, the social insurance program
that consists of retirement, disability and survivor benefits. To find out more about this program, visit the `SSA homepage <https://www.ssa.gov/about-ssa>`_.

Pseudopeople can generate a simulated version of a subset of the administrative data that would be recorded by SSA. Currently, the simulated
SSA data includes records of SSA creation and dates of death.

The following simulant attributes are included in this dataset:

.. list-table:: **Simulant attributes**
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
     -      
   * - Date of event
     - :code:`event_date`
     - Formatted as YYYY-MM-DD.  
   * - Type of event
     - :code:`event_type`
     - Possible values are "Creation" and "Death". 


Tax W-2 & 1099
--------------

The following simulant attributes are included in these datasets:

.. list-table:: **Simulant attributes**
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

Tax 1040
--------