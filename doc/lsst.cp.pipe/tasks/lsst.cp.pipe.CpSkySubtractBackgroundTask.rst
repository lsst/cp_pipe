.. lsst-task-topic:: lsst.cp.pipe.cpSkyTask.CpSkySubtractBackground

#######################
CpSkySubtractBackground
#######################

``CpSkySubtractBackground`` subtracts the scaled full-focal plane model created by :lsst-task:`~lsst.cp.pipe.CpSkyScaleMeasureTask` from the per-detector images created by :lsst-task:`~lsst.cp.pipe.CpSkyImageTask`.

.. _lsst.cp.pipe.cpSkyTask.CpSkySubtractBackground-processing-summary:

Processing summary
==================

``CpSkySubtractBackground`` runs these operations:

#. Subtract the scaled focal-plane model from the per-detector image.
#. Remeasure the residual background.

.. _lsst.cp.pipe.cpSkyTask.CpSkySubtractBackground-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpSkyTask.CpSkySubtractBackground

.. _lsst.cp.pipe.cpSkyTask.CpSkySubtractBackground-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpSkyTask.CpSkySubtractBackground

.. _lsst.cp.pipe.cpSkyTask.CpSkySubtractBackground-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpSkyTask.CpSkySubtractBackground
