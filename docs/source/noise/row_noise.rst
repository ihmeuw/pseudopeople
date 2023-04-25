.. _row_noise:

===============
Row-based Noise
===============

Row-based noise operates on one row of data at a time, for example by omitting
or duplicating entire rows.

Omit a row
----------

Sometimes an entire record may be missing from a dataset where one would
normally expect to find it. For example, a WIC record could be missing by
mistake because of a clerical error, or someone's tax record could be missing
because they didn't file their taxes on time.

This noise type is called :code:`omit_row` in the configuration. It takes one
parameter:

.. list-table:: Parameters to the omit_row noise type
  :widths: 1 5 1
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`row_probability`
    - The probability that a row is missing from the dataset.
    - 0.01 (1%)

When applying :code:`omit_row` noise, each row of data is selected for omission
independently with probability :code:`row_probability`.
