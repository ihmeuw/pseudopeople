.. _column_noise:

==================
Column-based Noise
==================

Column-based noise operates on one column of data at a time,
introducing errors to individual cells within the column.

In pseudopeople, column-based noise is currently performed independently
between columns.
For example, a simulant making a typo in their first name on the decennial Census
doesn't make them any more likely to make a typo in their last name as well,
even though in real life we might expect these things to be related.

Column-based noise is also unrelated to attributes of the simulant or record.
For example, a simulant who is 20 is just as likely to misreport their age as
a simulant who is 75.

These are both areas where we may add more complexity in future versions of pseudopeople.

Types of column-based noise:

.. contents::
   :depth: 2
   :local:

Leave a field blank
-------------------

Often some of the data in certain columns of a dataset will be missing.
This could be because the input for that field was left blank, an answer was refused,
or the answer was illegible or unintelligible.
When this type of noise occurs, pseudopeople will replace the value in the relevant cell with
:code:`numpy.nan` to indicate that the value is missing.

This noise type is called :code:`leave_blank` in the configuration. It takes one parameter:

.. list-table:: Parameters to the leave_blank noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that a cell in the column being configured is blank.
    - 0.01 (1%)

Choose the wrong option
-----------------------

If a question on a survey or administrative form provides a list of options,
respondents may sometimes choose the wrong option, either intentionally or by mistake.
pseudopeople simulates this type of noise by sometimes selecting an incorrect option
for columns that would have a list of options.
All wrong options are equally likely.
The possible values to select from depend on the column:
the :ref:`Datasets page <datasets_main>` lists them for each applicable column in pseudopeople's datasets.

This noise type is called :code:`choose_wrong_option` in the configuration.
It takes one parameter:

.. list-table:: Parameters to the choose_wrong_option noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that, for a cell in the column being configured, the wrong option is chosen.
    - 0.01 (1%)

Write the wrong ZIP code digits
-------------------------------

When reporting a ZIP code on a survey or form, people may misremember or misreport
the digits.
They are probably more likely to do this for the last few digits (which identify
the small, specific area) than the first few (which will be the same over a larger area).

This noise type is called :code:`write_wrong_zipcode_digits` in the configuration.
It takes two parameters:

.. list-table:: Parameters to the write_wrong_zipcode_digits noise type
  :widths: 1 5 3
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability of a cell being *considered* to have this noise type.
      One way to think about this is the probability that a ZIP code is reported by someone who isn't sure of their ZIP code.
      Whether or not there are actually any errors depends on the next parameter.
    - 0.01 (1%)
  * - :code:`digit_probabilities`
    - A list of five probabilities, one for each digit in a (5-digit) ZIP code.
      The first value in this list is the probability that the first digit of the ZIP code will be wrong
      **given that the cell is being considered for this noise type**.
      The second value in the list is the corresponding probability for the second digit, and so on.
    - [0.04, 0.04, 0.20, 0.36, 0.36]

Misreport age
-------------

When someone reports their age in years, or especially when someone reports the age of someone else such as a family member,
they may not get the value exactly right.
When this noise type occurs, the reported age is off by some amount, for example a year or two older or younger than the
person actually is.

This noise type is called :code:`misreport_age` in the configuration.
It takes two parameters:

.. list-table:: Parameters to the misreport_age noise type
  :widths: 1 5 3
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability of each age value being misreported.
    - 0.01 (1%)
  * - :code:`possible_age_differences`
    - One of two options:

        * A list of possible differences to add to the true age to get the misreported age.
          A negative number means that the reported age is too young, while a positive number means it is too old.
          Each difference is equally likely.
        * A dictionary mapping from possible differences to the corresponding probabilities of those differences.
          This is like the list option except that it allows some age differences to be more likely than others.
          The probabilities must add up to 1.
      
      Zero (no change) is not allowed as a possible difference.
    - {-2: 0.1, -1: 0.4, +1: 0.4, +2: 0.1}

We assume that age would never be incorrectly reported as a negative number.
In rare cases where applying the configured difference value would result in a negative age, we reflect this
age back to positive (e.g. -2 becomes 2).
This means there is still a spread of errors (they don't "bunch up" at zero).
If this reflection would cause the age to be correct, we instead make the reported age one year younger than
the true age.

Write the wrong digits
----------------------

Use a fake name
---------------

Make typos
----------
