.. _datasets_main:

========
Datasets
========
Here we cover the realistic simulated datasets, which are analogous to 'real world' administrative records such as tax documents
and census surveys, that users can generate using :code:`pseudopeople` for developing and testing Entity Resolution algorithms 
and software.

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
   * - | American Communities Survey (ACS)
   * - | Current Population Survey (CPS)
   * - | Women, Infants, and Children (WIC) Administrative Data
   * - | Social Security Administration (SSA) Data
   * - | Tax W2 and 1099 Forms
   * - | Tax 1040 Form


US Decennial Census
-------------------
The Decennial Census dataset is a simulated enumeration of the US Census Bureau's Decennial Census Survey. The years
that have been simulated are 2020, 2030, and 2040.

The following simulant attributes are included in this dataset:

.. list-table:: **Simulant attributes**
   :header-rows: 1

   * - Attribute Name
     - Column Name    
   * - Unique simulant ID
     - 
   * - First name
     - :code:`first_name`
   * - Middle initial
     - :code:`middle_initial`
   * - Last name
     - :code:`last_name`
   * - Age
     - :code:`age`  
   * - Date of birth
     - :code:`date_of_birth`
   * - Physical address street number
     - :code:`street_number`
   * - Physical address street name
     - :code:`street_name`
   * - Physical address unit
     - :code:`unit_number`
   * - Physical address city
     - :code:`city`    
   * - Physical address state
     - :code:`state`  
   * - Physical address ZIP code
     - :code:`zipcode`
   * - Relationship to person 1 (head of household)
     - :code:`relationship_to_household_head` 
   * - Sex (binary; 'male' or 'female')
     - :code:`sex`  
   * - Race/ethnicity
     - :code:`race_ethnicity` 

Household Surveys: ACS and CPS
------------------------------
There are two simulated household survey datasets that can be used: the American
Communities Survey (ACS) and the Current Population Survey (CPS). 

.. list-table:: **Simulant attributes**
   :header-rows: 1

   * - Attribute Name
     - Column Name
   * - Unique simulant ID
     - 
   * - Household ID 
     -  
   * - First name
     - :code:`first_name`
   * - Middle initial
     - :code:`middle_initial`
   * - Last name
     - :code:`last_name`
   * - Age
     - :code:`age`  
   * - Date of birth
     - :code:`date_of_birth`
   * - Physical address street number
     - :code:`street_number`
   * - Physical address street name
     - :code:`street_name`
   * - Physical address unit
     - :code:`unit_number`
   * - Physical address city
     - :code:`city`    
   * - Physical address state
     - :code:`state`  
   * - Physical address ZIP code
     - :code:`zipcode`
   * - Relationship to person 1 (head of household)
     - :code:`relationship_to_household_head` 
   * - Sex (binary; 'male' or 'female')
     - :code:`sex`  
   * - Race/ethnicity
     - :code:`race_ethnicity` 

WIC
---


Social Security
---------------


Tax W-2 & 1099
--------------


Tax 1040
--------