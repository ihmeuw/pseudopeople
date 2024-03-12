.. _row_noise:

===============
Row-based Noise
===============

Row-based noise operates on one row of data at a time, for example by omitting
or duplicating entire rows.

Types of row-based noise:

.. contents::
   :local:


Duplicate with guardian
-----------------------

A known challenge in entity resolution is people being reported multiple
times at different addresses. This can occur when family structures are
complex and children spend time at multiple households. A related
challenge occurs with college students, who are sometimes counted both at their
university and at their guardian’s home address.


To simulate such challenges, pseudopeople can apply this type of duplication to two mutually exclusive categories of
simulants based on age and GQ status: Simulants younger than 18 and not
in GQ and simulants under 24 and in college GQ.

For each of the two categories of simulants, a maximum duplication rate will
be calculated based on those who have a guardian living at a different address.
Most simulants in college GQ will have a guardian at a
different address, but most simulants younger than 18 will not.
If you as the user select a duplication rate that is higher than the 
calculated maximum rate, you will see a warning that 
the requested rate is greater than the maximum possible.

This noise type is called :code:`duplicate_with_guardian` in the configuration. 
It takes two parameters:

.. list-table:: Parameters to the duplicate_with_guardian noise type
  :widths: 1 5 3
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`row_probability_in_households_under_18`
    - The probability that a simulant under 18 in a household is recorded twice.
    - 0.02 (2%)
  * - :code:`row_probability_in_college_group_quarters_under_24`
    - The probability that a simulant under 24 in college GQ is recorded twice.
    - 0.05 (5%)

.. _do_not_respond:

Do not respond
--------------

Sometimes people don't respond to a census or survey questionnaire and are
unable to be reached by other means such as telephone or personal visit. For the
Decennial Census and household surveys such as the ACS and CPS, people are found
to respond at different rates depending on demographics such as age, sex, and
race or ethnicity.

For each demographic subgroup in pseudopeople, we assumed the nonresponse rate
in the Decennial Census was equal to the *net* rate of undercount (ignoring
duplication) estimated in [Census_PES]_. Net undercount effects of age/sex were
combined additively with the effects of race/ethnicity, and demographic
subgroups with resulting net overcounts (negative nonresponse rates) were given
a nonresponse rate of 0. We assumed nonresponse in the ACS was the same as in
the Decennial Census, since these are conducted similarly. By contrast, the CPS
uses only phone calls and personal visits but not mail/online questionnaires.
Thus we assumed that CPS had the same nonresponse pattern as the Decennial
Census, but with a constant 27.6% added to the nonresponse rates due to this
survey having fewer contact modes, as that was the nonresponse rate for CPS
reported for July 2022 in [Response_Rates_BLS]_.

To simulate nonresponse bias in the Decennial Census and the ACS or CPS, the
user can choose an overall rate of nonresponse, and pseudopeople will scale the
nonresponse rates for different demographic subgroups so that the overall
average rate approximately matches the target. The default overall rates were
calculated from our simulated population after applying the nonresponse rates
derived from [Census_PES]_ and [Response_Rates_BLS]_ to each demographic
subgroup as described above.

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

.. [Census_PES] Bureau, US Census. March 10, 2022. “Detailed Coverage Estimates for the 2020 Census Released Today.” Census.Gov. Accessed September 29, 2022. https://www.census.gov/library/stories/2022/03/who-was-undercounted-overcounted-in-2020-census.html.

.. [Response_Rates_BLS] “Household and Establishment Survey Response Rates: U.S. Bureau of Labor Statistics.” n.d. Accessed October 11, 2022. https://www.bls.gov/osmr/response-rates/home.htm.


Omit a row
----------

Sometimes an entire record may be missing from a dataset where one would
normally expect to find it. For example, a WIC record could be missing by
mistake because of an administrative error, or someone's tax record could be
missing because they didn't file their taxes on time.

This noise type is called :code:`omit_row` in the configuration. It takes one
parameter:

.. list-table:: Parameters to the omit_row noise type
  :widths: 1 5 3
  :header-rows: 1

  * - Parameter
    - Description
    - Default
  * - :code:`row_probability`
    - The probability that a row is missing from the dataset.
    - * 0.005 (0.5%) for WIC and tax forms W2 and 1099
      * 0.0 (0%) for other datasets

When applying :code:`omit_row` noise, each row of data is selected for omission
independently with probability :code:`row_probability`.
