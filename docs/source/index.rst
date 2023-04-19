============
Pseudopeople
============

Pseudopeople is a Python package that generates realistic simulated data about a
fictional United States population, for use in testing record linkage (entity resolution)
methods.

| 🙈 **Simulated**: These are made-up people! No need to worry about confidentiality.
| 📝 **Versatile**: Generate multiple datasets about the same population: censuses, surveys, and administrative records.
| ✔️ **Verifiable**: Ground-truth unique identifiers are present in every dataset for checking link correctness.
| ⚙️ **Customizable**: Configure the levels of noise in each dataset.
| 💪 **Full-scale**: Supports generating datasets at the size of the real-life US population.

Pseudopeople is currently in an alpha release.
Things are still in flux!
If you notice any issues, please let us know `on GitHub <https://github.com/ihmeuw/pseudopeople/issues>`_.

Quickstart
----------

First, install Pseudopeople with pip:

.. highlight:: console

::

   $ pip install pseudopeople

Or, you can install from source on `the Pseudopeople GitHub repository <https://github.com/ihmeuw/pseudopeople>`_.

Then, generate a small-scale simulated decennial census:

::

   $ python

.. highlight:: pycon

::

   >>> import pseudopeople as psp
   >>> census = psp.generate_decennial_census()
   >>> census
        simulant_id first_name middle_initial  last_name age date_of_birth street_number           street_name unit_number     city state zipcode relation_to_household_head     sex race_ethnicity
   0            0_2    Melanie              L     Herrod  26    08/05/1993         10233  north burgher avenue              Anytown    US   00000           Reference person  Female          White
   1            0_3     Jordan              C     Herrod  26    12/29/1993         10233  north burgher avenue              Anytown    US   00000             Other relative  Female          White
   2          0_923       John              E      Davis  77    06/29/1942       147-153          browning ave              Anytown    US   00000           Reference person    Male          Black
   3         0_2641     Sharon              T    Plummer  59    10/10/1960           107           stallion st              Anytown    US   00000           Reference person  Female          White
   4         0_2801     Ronnie              A     Yoakum  73    12/05/1946           214           s vine lane              Anytown    US   00000           Reference person    Male          White
   ...          ...        ...            ...        ...  ..           ...           ...                   ...         ...      ...   ...     ...                        ...     ...            ...
   9639     0_19008      James              G    Johnson  56    06/12/1963           691        sunny crest ln              Anytown    US   00000           Reference person    Male          Black
   9640     0_20161   Nannette              D    Mckenna  61    11/09/1958          6132                n pine              Anytown    US   00000           Reference person  Female          White
   9641     0_20162    Cynthia              L    Mckenna  65    01/20/1955          6132                n pine              Anytown    US   00000            Same-sex spouse  Female          White
   9642     0_19669     Marcus              A  Underwood  59    10/06/1960          1724              lodi way              Anytown    US   00000           Reference person    Male            NaN
   9643     0_20160      Kelly              A     Parris  26    10/24/1993          2203           blume st ne              Anytown    US   00000           Reference person  Female          White

   [9644 rows x 15 columns]

And W-2 and 1099 tax forms from the same fake population:

::

   >>> taxes = psp.generate_taxes_w2_and_1099()
   >>> taxes
        simulant_id first_name middle_initial  last_name age date_of_birth mailing_address_street_number  ... employer_street_number employer_street_name employer_unit_number employer_city employer_state employer_zipcode tax_form
   0            0_4    Michael              M      Ticas  37    03/13/1983                          1312  ...                      e              ince dr                            Anytown             US            00000       W2
   1            0_5   Michelle              M      Ticas  39    08/10/1981                          1312  ...                 264-12          hallmont dr                            Anytown             US            00000       W2
   2            0_5   Michelle              M      Ticas  39    08/10/1981                          1312  ...                                 ne 39th ave                            Anytown             US            00000       W2
   3         0_5621    Jeffrey              S  Contreras  50    07/26/1970                                ...                     38         mckenzie hwy                            Anytown             US            00000       W2
   4         0_5623     Gloria              A  Contreras  47    07/23/1973                                ...                   2916            4th ave w                            Anytown             US            00000       W2
   ...          ...        ...            ...        ...  ..           ...                           ...  ...                    ...                  ...                  ...           ...            ...              ...      ...
   9083     0_18936          L              J     Molina  49    09/12/1971                          1070  ...                                barge roa sw                            Anytown             US            00000       W2
   9084     0_18936     Robert              J     Molina  49    09/12/1971                          1070  ...                      e              ince dr                            Anytown             US            00000       W2
   9085     0_18937     Angela              M     Molina  40    11/24/1980                          1070  ...                    210           regatta dr                            Anytown             US            00000     1099
   9086     0_18937     Angela              M     Molina  40    11/24/1980                          1070  ...                   2916            4th ave w                            Anytown             US            00000     1099
   9087     0_18938      Anais              K     Molina  19    11/14/2001                          1070  ...                    210           regatta dr                            Anytown             US            00000       W2

   [9088 rows x 24 columns]

The simulated people in these datasets are called "simulants."
Both datasets have a :code:`simulant_id` column that uniquely identifies an individual.
This is our truth deck, which we wouldn't have for a real linkage task.

::

   >>> true_matches = len(set(census.simulant_id) & set(taxes.simulant_id))
   >>> print(f"There are {true_matches:,} true matches to find between these datasets!")
   There are 6,181 true matches to find between these datasets!

Now, see how many your record linkage method can find -- without access to the truth deck, of course!

.. toctree::
   :hidden:
   :maxdepth: 2

   datasets/index
   noise_functions/index
   tutorials/index
   concepts/index
   api_reference/index
   glossary
