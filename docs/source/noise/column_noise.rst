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

Use a fake name
---------------

Make typos
----------

Typos occur in survey and administrative datasets when someone -- a survey respondent, a canvasser,
or someone entering their own information on a form -- types a value incorrectly.

Currently, pseudopeople implements two kinds of typos: inserting extra characters
directly preceding characters that are adjacent on a keyboard, or replacing a character with one that is adjacent.
When pseudopeople introduces typos, 10% of them are inserted characters, while the other 90% are replaced characters.
This is currently not configurable.

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
      One way to think about this is the probability of an typo on any given character when the value is being typed carelessly.
    - 0.1 (10%)