.. _input_data_main:

==========
Input Data
==========

pseudpoeple leverages the power of the `Vivarium <https://vivarium.readthedocs.io/en/latest/>`_ microsimulation platform to incorporate real, publicly-accessible data about the US population. The input data for pseudopeople is the output of a Vivarium simulation and must be in a specific format for the dataset generating functions to work.
There are currently three collections of pseudopeople input data:

- Sample data (a fictional population of ~10,000 simulants living in Anytown, US, included with the pseudopeople package)
- Rhode Island (a fictional population of ~1,000,000 simulants living in a simulated state of Rhode Island)
- United States (a fictional population of ~330,000,000 simulants living throughout a simulated United States)

A collection of small-scale sample datasets is included with the software, and pseudopeople uses this sample data by default unless an explicit path to another directory containing pseudopeople input data is specified.
There is a data access request process to obtain larger-scale data that can be used with pseudopeople. Currently
