.. py:currentmodule:: lsst.cp.pipe

.. _lsst.cp.pipe:

############
lsst.cp.pipe
############

.. This module is used to produce the calibration products required to perform instrument signal removal for a camera. Producing master calibration frames, generating bad pixel/column masks, calculating photon transfer curves (and the child products of that, linearity models and brighter-fatter kernels), and measuring crosstalk coefficients.

.. _lsst.cp.pipe-using:

Using lsst.cp.pipe
==================

.. toctree::
   :maxdepth: 1

   constructing-calibrations

.. _lsst.cp.pipe-contributing:

Contributing
============

``lsst.cp.pipe`` is developed at https://github.com/lsst/cp_pipe.
You can find Jira issues for this module under the `cp_pipe <https://jira.lsstcorp.org/issues/?jql=project%20%3D%20DM%20AND%20component%20%3D%20cp_pipe>`_ component.

.. If there are topics related to developing this module (rather than using it), link to this from a toctree placed here.

.. .. toctree::
..    :maxdepth: 1

Task reference
==============

.. _lsst.cp.pipe-pipeline-tasks:

Pipeline tasks
--------------

.. lsst-pipelinetasks::
   :root: lsst.cp.pipe

.. _lsst.cp.pipe-command-line-tasks:

Command-line tasks
------------------

.. lsst-cmdlinetasks::
   :root: lsst.cp.pipe

Python API reference
====================

.. automodapi:: lsst.cp.pipe
   :no-main-docstr:
   :no-inheritance-diagram:
