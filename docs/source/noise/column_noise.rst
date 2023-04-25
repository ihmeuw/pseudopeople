.. _column_noise:

==================
Column-based Noise
==================

Leave a field blank
-------------------

Choose the wrong option
-----------------------

Write the wrong zipcode digits
------------------------------

Misreport age
-------------

Write the wrong digits
----------------------

Sometimes people may write the wrong number for numeric data such as a street
number, date, or social security number. This could be intentional or
accidental. pseudopeople simulates this type of noise in fields that include
numbers by randomly replacing some digits with different digits selected
uniformly at random.

This noise type is called :code:`write_wrong_digits` in the configuration.
It takes two parameters:

.. list-table:: Parameters to the write_wrong_digits noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that any given cell in the column will be selected to be eligible for this type of noise.
    - 0.01 (1%)
  * - :code:`token_probability`
    - The conditional probability, given that a numeric cell has been selected for noise eligibility, that any given digit in the true number will be replaced by a different digit.
    - 0.1 (10%)


Use a fake name
---------------

Make typos
----------
