.. _noise_functions_concept:

=================
 Noise Functions
=================

.. contents::
   :depth: 2
   :local:
   :backlinks: none




What is a noise function?
-------------------------

A noise function is ultimately where the configuration (add link) provided is
applied to the raw data which is then noised or altered and returned to the user
in a state where real world data error have been added to each dataset (add link).
Noise functions will be applied to datasets by column or by row.  There are
several noise functions that are applied to the raw data which include:

.. list-table:: **Noise Functions**
   :header-rows: 1
   :widths: 20

   * - Name
   * - | Omission
   * - | Duplications
   * - | Missing data
   * - | Incorrect selection
   * - | Copy from within household
   * - | Month and day swaps (applies to dates only)
   * - | Zip Code Miswriting (applies to Zip Code only)
   * - | Age Miswriting (applies to age only)
   * - | Numeric miswriting
   * - | Nicknames
   * - | Fake names
   * - | Phonetic errors
   * - | OCR (optical character recognition)
   * - | Typographic
