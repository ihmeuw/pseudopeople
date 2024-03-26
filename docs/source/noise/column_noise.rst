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

Borrow a social security number
-------------------------------

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
or the answer was illegible or unintelligible. To simulate this type of noise, pseudopeople will 
replace the value in the relevant cell with :code:`numpy.nan` to indicate that the value is missing. 

It is important to note, however, that 
some columns in the generated data may contain missing values, even if no noise has been added to the data,
simply because the column is not applicable to every row.
Columns that may have missing values regardless of noise include unit number, street number, and any columns pertaining
to spouse or dependents in the 1040 tax dataset, for example. In these cases where fields are blank even without noise,
missing values are also represented by :code:`numpy.nan`.

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

Copy from household member
--------------------------

When responding to a survey or filling out a form, someone might accidentally or
purposely answer a question about one household member with information about a
different household member. To capture this type of error, pseudopeople can fill
in certain fields about a simulant with values from a different member of the
simulant's household, chosen at random. This type of noise can be applied to
ages, dates of birth, and social security numbers.

Note that simulants who live in group quarters or who live alone are not
eligible for this type of noise, so for each dataset, there is some maximum
fraction of rows to which "copy from household member" noise can be applied. If
the user requests a cell probability that is larger than what's possible,
pseudopeople will add noise to the maximum possible number of rows.

This noise type is called :code:`copy_from_household_member` in the configuration. It takes one parameter:

.. list-table:: Parameters to the copy_from_household_member noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that, for a cell in the column being configured, the cell's value is replaced by the corresponding value from a household member.
    - 0.01 (1%)

**Note:** The default value of 0.01 applies to most datasets. However, the
default value is 0.0 for the SSN column in the W2 & 1099 dataset since SSNs are
already subject to "borrow a social security number" noise in that dataset, and
is also 0.0 for the SSN column in the SSA dataset because that column has no
noise by default.

.. _use_a_nickname:

Use a nickname
---------------

Many people, when filling out forms or survey answers, choose to use nicknames instead of their legal names.
A common example is an Alexander who chooses to go by Alex.

The "Use a nickname" noise type in pseudopeople simulates these kinds of responses for first and middle names. In order
to do this, we used a list of 1,080 names and their relevant nicknames, from a project by Old Dominion
University's Web Science and Digital Libraries Research Group. You can read more about the list of nicknames
in the group's `GitHub repository <https://github.com/carltonnorthern/nicknames>`_.

Instead of the person's legal name, pseudopeople selects the subset of simulated individuals who are eligible
for a nickname (i.e., those whose legal first or middle name is included in the nicknames list detailed above), then replaces
each selected simulant's first name with any of the nicknames included in the csv file.

This noise type is called :code:`use_nickname` in the configuration. It takes one parameter:

.. list-table:: Parameters to the use_nickname noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability that, for a cell in the :code:`first_name` column, a nickname is recorded.
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
`NORC assessment of the Census Bureau's Person Identification Validation System <https://web.archive.org/web/20230705005935/https://www.norc.org/content/dam/norc-org/pdfs/PVS%20Assessment%20Report%20FINAL%20JULY%202011.pdf>`_.

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

Swap month and day
------------------

Swap month and day is a noise type that only applies to dates. It occurs when
someone swaps the month and day to be in the incorrect position (e.g., December
8, 2022 would be listed in MM/DD/YYYY format as 08/12/2022).

This noise type is called :code:`swap_month_and_day` in the configuration. It
takes one parameter:

.. list-table:: Parameters to the swap_month_and_day noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability of a cell date having its month and day swapped.
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

Make phonetic errors
--------------------
A phonetic error occurs when a character is misheard. For instance, this could happen with similar sounding letters when spoken (like ‘t’ and ‘d’) or letters that make the same sounds within a word (like ‘o’ and ‘ou’).

pseudopeople defines the possible phonetic substitutions using `this file <https://github.com/ihmeuw/pseudopeople/blob/develop/src/pseudopeople/data/phonetic_variations.csv>`_, which was produced by the `GeCO project <https://dl.acm.org/doi/10.1145/2505515.2508207>`_.

This noise type is called :code:`make_phonetic_errors` in the configuration. It takes two parameters:

.. list-table:: Parameters to the make_phonetic_errors noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability of a cell being *considered* to have this noise type.
      One way to think about this is the probability that a string is transcribed by an error-prone program or human transcriber.
      Whether or not there are actually any errors depends on the next parameter.
    - 0.01 (1%)
  * - :code:`token_probability`
    - The probability of each corruption-eligible token being misheard
      **given that the cell is being considered for this noise type**.
      One way to think about this is the probability of a phonetic error on any given corruption-eligible token when the transcriber is error-prone.
    - 0.1 (10%)

Make optical character recognition (OCR) errors
--------------------------------------------------

An optical character recognition (OCR) error is when a string is misread for another string that is visually similar. Some common examples are
‘S’ instead of ‘5’ and ‘m’ instead of ‘iii’.

pseudopeople defines the possible OCR substitutions using `this CSV file <https://github.com/ihmeuw/pseudopeople/blob/develop/src/pseudopeople/data/ocr_errors.csv>`_, which was produced by the `GeCO project <https://dl.acm.org/doi/10.1145/2505515.2508207>`_. In the file, the first column is the real string (which we call a "token") and the second column is what it could be misread as (a "corruption").
The same token can be associated with multiple corruptions.

To implement this, we first select the rows to noise, as in other noise types.
For those rows, each corruption-eligible token in the relevant string is selected to be corrupted or not,
according to the token noise probability.
Each token selected for corruption is replaced with its corruption according to the above CSV file
(choosing uniformly at random in the case of multiple corruption options for a single token),
**unless a token with any overlapping characters (in the original string) has already been corrupted**.

.. note::
  Tokens are corrupted in the order of the location of their first character in the original string, from beginning to end,
  breaking ties (e.g. 'l' and 'l>' are both corruption-eligible tokens and may start on the same 'l') by corrupting longer tokens first.
  Note that in an example :code:`abcd` where :code:`ab`, :code:`bc`, **and** :code:`cd` have **all** been selected to be corrupted,
  the corruption of :code:`ab` prevents the corruption of :code:`bc` from occurring, which then allows :code:`cd` to be corrupted
  even though it overlapped with :code:`bc`.

This noise type is called :code:`make_ocr_errors` in the configuration. It takes two parameters:

.. list-table:: Parameters to the make_ocr_errors noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`cell_probability`
    - The probability of a cell being *considered* to have this noise type.
      One way to think about this is the probability that a string is read by an inaccurate OCR program or human reader.
      Whether or not there are actually any errors depends on the next parameter.
    - 0.01 (1%)
  * - :code:`token_probability`
    - The probability of each corruption-eligible token being misread
      **given that the cell is being considered for this noise type**.
      One way to think about this is the probability of an OCR error on any given corruption-eligible token when a string is being read inaccurately.
    - 0.1 (10%)

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

.. list-table:: Parameters to the make_typos noise type
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
