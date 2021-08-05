.. lsst-task-topic:: lsst.cp.pipe.CrosstalkSolveTask

##################
CrosstalkSolveTask
##################

``CrosstalkSolveTask`` fits the crosstalk coefficients based on the full set of input flux ratios.

.. _lsst.cp.pipe.CrosstalkSolveTask-processing-summary:

Processing summary
==================

``CrosstalkSolveTask`` runs these operations:

#. Combines all of the individual exposure flux ratios measured by :lsst-task:`~lsst.cp.pipe.CrosstalkExtractTask` into one set.
#. Fits each pair of source-target amplifier pairs to find the clipped mean value.
#. Optionally prunes coefficients that have coefficients that are statistically consistent with the flux ratio noise.

.. _lsst.cp.pipe.CrosstalkSolveTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CrosstalkSolveTask

.. _lsst.cp.pipe.CrosstalkSolveTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.CrosstalkSolveTask

.. _lsst.cp.pipe.CrosstalkSolveTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CrosstalkSolveTask

.. _lsst.cp.pipe.CrosstalkSolveTask-debug:

Debugging
=========

reduce
    Display a histogram of the combined ratio measurements for a pair of source/target amplifiers from all input exposures/detectors (`bool`)?

measure
    Display the CDF of the combined ratio measurements for a pair of source/target amplifiers from the final set of clipped input ratios (`bool`)?
