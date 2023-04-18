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

Pseudopeople applies the different types of row-based and column-based noise
in the following order so as to mimic the data generation process by which a
real dataset might have errors introduced into it.
