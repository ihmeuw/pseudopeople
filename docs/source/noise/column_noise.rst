.. _column_noise:

==================
Column-based Noise
==================

Column-based noise introduces errors in individual fields within a row of data.
The types implemented in pseudopeople are listed below, along with instructions
about how to configure them.

.. contents::
   :depth: 2
   :local:

Leave a field blank
-------------------

When this noise type occurs, whatever value was supposed to be collected is missing for this row.
This could be because the input was left blank, an answer was refused,
or the answer was illegible or unintelligible.
pseudopeople will output a :code:`numpy.nan` value in
the relevant cell.

This noise type is called :code:`leave_blank` in the configuration. It takes one parameter:

.. list-table:: Parameters to the :code:`leave_blank` noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that a cell in the column being configured is blank.
      This probability is the same for all cells in a column;
      missingness is not correlated with any attributes.
    - 0.01 (1%)

Choose the wrong option
-----------------------

Some survey questions or administrative forms ask the respondent to choose
between a list of options.
This noise type occurs when the wrong option is chosen by mistake.
The options available depend on the column; see the :ref:`datasets page <datasets_main>` for
lists of options that correspond to each column.
All wrong options are equally likely.

This noise type is called :code:`choose_wrong_option` in the configuration.
It takes one parameter:

.. list-table:: Parameters to the :code:`choose_wrong_option` noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that, for a cell in the column being configured, the wrong option is chosen.
      The probability of this noise type is the same for all cells in a column;
      it is not correlated with any attributes.
    - 0.01 (1%)

Write the wrong ZIP code digits
-------------------------------

When reporting a ZIP code on a survey or form, people may misremember or misreport
the digits.
They are probably more likely to do this for the last few digits (which identify
the small, specific area) than the first few (which will be the same over a larger area).

This noise type is called :code:`write_wrong_zipcode_digits` in the configuration.
It takes two parameters:

.. list-table:: Parameters to the :code:`write_wrong_zipcode_digits` noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability of a cell being *considered* to have this noise type.
      One way to think about this is the probability that a ZIP code is reported by someone who isn't sure of their ZIP code.
      This probability is the same for all cells in a column; it is not correlated with any attributes.
      Whether or not there are actually any errors depends on the next parameter.
    - 0.01 (1%)
  * - :code:`digit_probabilities`
    - A list of probabilities, one for each digit in a (5-digit) ZIP code.
      The first item in this list is the probability **in a cell considered for this noise type** that the first digit of the ZIP
      code will be wrong, the second item is the same probability for the second digit, and so on.
    - [0.2, 0.2, 0.2, 0.36, 0.36]

Misreport age
-------------

When someone reports their age in years, or especially when someone reports the age of someone else such as a family member,
they may not get the value exactly right.
When this noise type occurs, the reported age is off by some amount, for example a year or two older or younger than the
person actually is.

This noise type is called :code:`misreport_age` in the configuration.
It takes two parameters:

.. list-table:: Parameters to the :code:`misreport_age` noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that, for an age value in the column being configured, the age is misreported.
      The probability of this noise type is the same for all cells in a column;
      it is not correlated with any attributes.
    - 0.01 (1%)
  * - :code:`possible_age_differences`
    - One of two options:

        * A list of possible differences to add to the true age to get the misreported age.
          A negative number means that the reported age is too young, while a positive number means it is too old.
          Each difference is equally likely.
        * A dictionary mapping from possible differences to the corresponding probabilities of those differences.
          This is like the list option except that it allows some differences to be more likely than others.
          The probabilities must add up to 1.
    - {-2: 0.1, -1: 0.4, +1: 0.4, +2: 0.1}

Write the wrong digits
----------------------

Use a fake name
---------------

Make typos
----------
