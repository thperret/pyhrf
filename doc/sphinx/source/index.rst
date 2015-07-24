
PyHRF: a python package to study hemodynamics in fMRI.
======================================================

------------------------------------------------------

Overview
--------

PyHRF is a set of tools for within-subject fMRI data analysis, focused on the characterization of the hemodynamics. 

Within the chain of fMRI data processing, these tools provide alternatives to the classical within-subject GLM estimation step. The inputs are preprocessed within-subject data and the outputs are statistical maps and/or fitted HRFs.

The package is mainly written in Python and provides the implementation of the two following methods:

      * **The joint-detection estimation (JDE)** approach, which divides the brain into functionnaly homogeneous regions and provides one HRF estimate per region as well as response levels specific to each voxel and each experimental condition. This method embeds a temporal regularization on the estimated HRFs and an adaptive spatial regularization on the response levels.

      * **The Regularized Finite Impulse Response (RFIR)** approach, which provides HRF estimates for each voxel and experimental conditions. This method embeds a temporal regularization on the HRF shapes, but proceeds independently across voxels (no spatial model).

See :ref:`introduction` for a more detailed overview.

To cite PyHRF and get a comprehensive description, please refer to `this paper <http://journal.frontiersin.org/Journal/10.3389/fnins.2014.00067/>`_:
    
    T. Vincent, S. Badillo, L. Risser, L. Chaari, C. Bakhous, F. Forbes and P.
    Ciuciu “Flexible multivariate hemodynamics fMRI data analyses and
    simulations with PyHRF” Font. Neurosci., vol. 8, no. 67, 10 April 2014.
|

.. Developpment status
.. -------------------

Site content:
-------------
    .. toctree::
       :maxdepth: 2     
    
       introduction.rst
       installation.rst
       manual.rst
       autodoc/pyhrf.rst

..       
    Indices and tables
    ==================
    
    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`

Licence and authors
-------------------

PyHRF is currently under the `CeCILL licence version 2 <http://www.cecill.info>`_. It is mainly developed at the `LNAO <http://www.lnao.fr>`_ (Neurospin, CEA) and the `MISTIS team <http://mistis.inrialpes.fr/>`_ (INRIA Rhones-Alpes).

Authors are:
         Thomas Vincent\ :sup:`(2)`, Philippe Ciuciu\ :sup:`(1)`, Lotfi Chaari\ :sup:`(2)`, Solveig Badillo\ :sup:`(1)`, Christine Bakhous\ :sup:`(2)`

         1. CEA/DSV/I2BM/Neurospin, LNAO, Gif-Sur-Yvette, France
         2. INRIA, MISTIS, Grenoble, France

Contacts
++++++++

thomas.tv.vincent@gmail.com, philippe.ciuciu@cea.fr        
