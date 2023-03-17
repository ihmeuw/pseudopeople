=======================
Installing Pseudopeople
=======================

.. contents::
   :depth: 1
   :local:
   :backlinks: none

.. highlight:: console

Overview
--------

Pseudopeople is written in `Python`__ and supports Python 3.7+.

__ http://docs.python-guide.org/en/latest/

.. _install-pypi:

Installation from PyPI
----------------------

Pseudopeople packages are published on the `Python Package Index
<https://pypi.org/project/pseudopeople/>`_. The preferred tool for installing
packages from *PyPI* is :command:`pip`.  This tool is provided with all modern
versions of Python

On Linux or MacOS, you should open your terminal and run the following command.

::

   $ pip install -U pseudopeople

On Windows, you should open *Command Prompt* and run the same command.

.. code-block:: doscon

   C:\> pip install -U pseudopeople


Installation from source
------------------------

You can install Pseudopeople directly from a clone of the `Git repository`__.
You can clone the repository locally and install from the local clone::

    $ git clone https://github.com/ihmeuw/pseudopeople.git
    $ cd pseudopeople
    $ pip install .

You can also install directly from the git repository with pip::

    $ pip install git+https://github.com/ihmeuw/pseudopeople.git

Additionally, you can download a snapshot of the Git repository in either
`tar.gz`__ or `zip`__ format.  Once downloaded and extracted, these can be
installed with :command:`pip` as above.

.. highlight:: default

__ https://github.com/ihmeuw/pseudopeople
__ https://github.com/ihmeuw/pseudopeople/archive/develop.tar.gz
__ https://github.com/ihmeuw/pseudopeople/archive/develop.zip
