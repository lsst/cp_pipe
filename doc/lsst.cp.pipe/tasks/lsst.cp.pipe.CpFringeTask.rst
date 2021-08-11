.. lsst-task-topic:: lsst.cp.pipe.cpFringeTask.CpFringeTask

############
CpFringeTask
############

``CpFringeTask`` preprocesses the input exposures to prepare them for combination into a master fringe calibration.

.. _lsst.cp.pipe.cpFringeTask.CpFringeTask-processing-summary:

Processing summary
==================

``CpFringeTask`` runs these operations:

#. Divides the input exposure by the measured background level, normalizing the image sky to 1.0.
#. Finds all sources above the masking threshold, and masks them

.. _lsst.cp.pipe.cpFringeTask.CpFringeTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpFringeTask.CpFringeTask

.. _lsst.cp.pipe.cpFringeTask.CpFringeTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpFringeTask.CpFringeTask

.. _lsst.cp.pipe.cpFringeTask.CpFringeTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpFringeTask.CpFringeTask
