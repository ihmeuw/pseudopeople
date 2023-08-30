.. _row_noise:

===============
Row-based Noise
===============

Row-based noise operates on one row of data at a time, for example by omitting
or duplicating entire rows.

Do not respond
--------------

Sometimes people don't respond to a census or survey questionnaire and are
unable to be reached by other means such as telephone or personal visit. For the
Decennial Census and household surveys (ACS and CPS), people are found to
respond at different rates depending on demographics (e.g., age, sex,
race/ethnicity) and mode of contact (e.g., mail/online, telephone, personal
visit). To simulate nonresponse bias in the Decennial Census and household
surveys, the user can choose an overall rate of nonresponse, and pseudopeople
will scale the nonresponse rates for different demographic subgroups to match
the overall target. The data sources used to inform differential nonresponse
rates and default overall nonresponse rates for the simulated Decennial Census
and household surveys are

* `Post-Enumeration Survey (PES) estimates of coverage error for the 2020 Census <https://www.census.gov/library/stories/2022/03/who-was-undercounted-overcounted-in-2020-census.html>`_
* `Response Profile of the 2005 ACS, by Geoffrey I. Jackson, US Census Bureau <https://www.fcsm.gov/assets/files/docs/2007FCSM_Jackson-III-C.pdf>`_
* `Household and Establishment Survey Response Rates: U.S. Bureau of Labor Statistics <https://www.bls.gov/osmr/response-rates/home.htm>`_

This noise type is called :code:`do_not_respond` in the configuration. It takes
one parameter:

.. list-table:: Parameters to the do_not_respond noise type
  :widths: 1 5 3
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`row_probability`
    - The probability that a simulant does not respond to the census or survey.
    - * 0.0145 (1.45%) for the Decennial Census and ACS
      * 0.2905 (29.05%) for CPS

Omit a row
----------

Sometimes an entire record may be missing from a dataset where one would
normally expect to find it. For example, a WIC record could be missing by
mistake because of an administrative error, or someone's tax record could be
missing because they didn't file their taxes on time.

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
