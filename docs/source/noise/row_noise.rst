.. _row_noise:

===============
Row-based Noise
===============

Row-based noise operates on one row of data at a time, for example by
introducing errors to certain cells within a row, or by omitting or duplicating
entire rows.

Omit a row
----------

Sometimes an entire record may be missing from a dataset where one would
normally expect to find it. For example, if someone didn't file their taxes on
time, then their tax record for that year would missing. Or perhaps a record is
missing by mistake because of a clerical error.

This noise type is called :code:`omit_row` in the configuration. It takes one parameter:

.. list-table:: Parameters to the omit_row noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`row_probability`
    - The probability that a row is missing from the dataset.
    - 0.01 (1%)
