.. _development

===========
Development
===========

This document is mainly addressed to pyhrf's developers.

Code 
####

You can find some guidelines on the `github's wiki <https://github.com/pyhrf/pyhrf/wiki>`_ :

    - `Code quality <https://github.com/pyhrf/pyhrf/wiki/Code-Quality>`_
    - `Unit tests <https://github.com/pyhrf/pyhrf/wiki/Unit-testing>`_
    - `Version number <https://github.com/pyhrf/pyhrf/wiki/Numbering>`_

Git
###

You can find some guidelines on the `github's wiki <https://github.com/pyhrf/pyhrf/wiki>`_ :

    - `Git workflow <https://github.com/pyhrf/pyhrf/wiki/Git-workflow>`_
    - `Merge and rebase <https://github.com/pyhrf/pyhrf/wiki/Merge-and-rebase>`_

For github's specific info check :

    - `Pull requests <https://github.com/pyhrf/pyhrf/wiki/Pull-Requests>`_
    - `Tickets <https://github.com/pyhrf/pyhrf/wiki/Tickets>`_

Documentation
#############

Make sure that you have installed latest sphinx version.

Autodoc
*******

To update the autodoc structure, `cd` to the source root folder and run::
    
    .. code:: bash
    
        $ rm -rf doc/sphinx/source/autodoc/*
        $ sphinx-apidoc -f -e -M -o doc/sphinx/source/autodoc/. ./pyhrf ./pyhrf/test ./pyhrf/validation

Docstrings
**********

Ensure that every modules/classes/methods/functions has its own doctring.
Some exceptions are for builtins methods (e.g.: `__init__`, ...)

For more informations, check `docstrings on wiki <https://github.com/pyhrf/pyhrf/wiki/Code-Quality#pep257-docstrings>`_
