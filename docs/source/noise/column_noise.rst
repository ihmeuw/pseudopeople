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

"Borrowed" SSN
--------------

The W-2 and 1099 tax forms require a Social Security Number (SSN).
Many people who are employed in the US do not have an SSN,
but they or their employer still file W-2 or 1099 forms, presumably using someone else's
SSN or a made-up SSN.

As a simple way to replicate this behavior, when a simulant without an SSN has a W-2 or 1099 filed,
pseudopeople uses an SSN borrowed from a randomly selected simulant in their household who does have one.
If there is nobody in their household with an SSN, a totally random SSN is created and used on the form.

This type of noise cannot be configured.
It is always present on all W-2 and 1099 forms about a simulant who does not have an SSN.

Leave a field blank
-------------------

Often some of the data in certain columns of a dataset will be missing.
This could be because the input for that field was left blank, an answer was refused,
or the answer was illegible or unintelligible.
To simulate this type of noise, pseudopeople will replace the value in the relevant cell with
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

.. _choose_the_wrong_option:

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

.. _use_a_fake_name:

Use a fake name
---------------

Sometimes when people respond to a survey or fill out a form, they don't want to share their personal information.
If the survey or form (whether online, on paper, or in person) requires a response, they might just make
something up.

The "Use a fake name" noise type in pseudopeople simulates these kinds of responses for first and last names.
Instead of the person's real name, pseudopeople records a randomly selected value from the
"List of First Names Considered Fake or Incomplete" (for first names) or the "List of Last Names Considered Fake or Incomplete" (for last names)
found in the
`NORC assessment of the Census Bureau's Person Identification Validation System <https://www.norc.org/Research/Projects/Pages/census-personal-validation-system-assessment-pvs.aspx>`_.

This noise type is called :code:`use_fake_name` in the configuration. It takes one parameter:

.. list-table:: Parameters to the use_fake_name noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that, for a cell in the column (either first or last name), a fake name is recorded.
    - 0.01 (1%)

Misreport age
-------------

When someone reports their age in years, or especially when someone reports the age of someone else such as a family member,
they may not get the value exactly right.
For this type of simulated noise, the reported age is off by some amount, for example a year or two older or younger than the
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
        * A dictionary, where the keys are the possible differences and
          the values are the probabilities of those differences.
          This is like the list option, except that it allows some age differences to be more likely than others.
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

Write the wrong ZIP code digits
-------------------------------

When reporting a ZIP code on a survey or form, people may misremember or misreport
the digits.
They are probably more likely to do this for the last few digits (which identify
the small, specific area) than the first few (which will be the same over a larger area).
The "Write the wrong ZIP code digits" noise type is just like "Write the wrong digits"
except that it can capture this difference between digits in different positions.
The ZIP code column uses this noise type instead of "Write the wrong digits" for this reason.

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

Make typos
----------

Typos occur in survey and administrative datasets when someone -- a survey respondent, a canvasser,
or someone entering their own information on a form -- types a value incorrectly.

Currently, pseudopeople implements two kinds of typos: inserting extra characters
directly preceding characters that are adjacent on a keyboard, or replacing a character with one that is adjacent.
When pseudopeople introduces typos, 10% of them are inserted characters, while the other 90% are replaced characters.
This is currently not configurable.
In either kind of typo, all adjacent characters are equally likely to be chosen.

To define "adjacent", we use a grid version of a QWERTY keyboard layout
(left-justified, which is not exactly accurate to most keyboards' half-key-offset layout) and accompanying number pad.
This approach is inspired by the GeCO project, with some changes to include capital letters and have a complete numberpad.
Two characters are considered adjacent if they are touching, either on a side or diagonally:

.. code-block:: text

  qwertyuiop
  asdfghjkl
  zxcvbnm

  QWERTYUIOP
  ASDFGHJKL
  ZXCVBNM

  789
  456
  123
  0

Note that there are empty lines above, which separate the parts.
Therefore, a number is never replaced by a letter (or vice versa), and a capital letter is never replaced by a lowercase letter (or vice versa).
There are currently no typos involving special characters.

This noise type is called :code:`make_typos` in the configuration. It takes two parameters:

.. list-table:: Parameters to the leave_blank noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability of a cell being *considered* to have this noise type.
      One way to think about this is the probability that a value is typed carelessly.
      Whether or not there are actually any errors depends on the next parameter.
    - 0.01 (1%)
  * - :code:`token_probability`
    - The probability of each character (which we call a "token") having a typo
      **given that the cell is being considered for this noise type**.
      One way to think about this is the probability of a typo on any given character when the value is being typed carelessly.
    - 0.1 (10%)