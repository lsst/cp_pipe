.. lsst-task-topic:: lsst.cp.pipe.cpFlatNormTask.CpFlatNormalizationTask

#######################
CpFlatNormalizationTask
#######################

``CpFlatNormalizationTask`` determines the scaling factor to apply to each exposure/detector set when constructing the final flat field.

.. _lsst.cp.pipe.cpFlatNormTask.CpFlatNormalizationTask-processing-summary:

Processing summary
==================

``CpFlatNormalizationTask`` runs these operations:

#. Combine the set of background measurements for all input exposures for all detectors into a matrix ``B[exposure, detector]``.
#. Iteratively solve for two vectors ``E[exposure]`` and ``G[detector]`` whose Cartesian product are the best fit to ``B[exposure, detector]``.

.. _lsst.cp.pipe.cpFlatNormTask.CpFlatNormalizationTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpFlatNormTask.CpFlatNormalizationTask

.. _lsst.cp.pipe.cpFlatNormTask.CpFlatNormalizationTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpFlatNormTask.CpFlatNormalizationTask

.. _lsst.cp.pipe.cpFlatNormTask.CpFlatNormalizationTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpFlatNormTask.CpFlatNormalizationTask
