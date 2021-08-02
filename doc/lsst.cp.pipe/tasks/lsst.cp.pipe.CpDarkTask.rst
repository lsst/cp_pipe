.. lsst-task-topic:: lsst.cp.pipe.cpDarkTask.CpDarkTask

##########
CpDarkTask
##########

``CpDarkTask`` preprocesses exposures after :lsst-task:`~lsst.ip.isr.IsrTask` and before the final dark combination.

.. _lsst.cp.pipe.cpDarkTask.CpDarkTask-processing-summary:

Processing summary
==================

``CpDarkTask`` runs these operations:

#. Identifies and masks cosmic rays.
#. Optionally grows the cosmic ray masks to ensure they do not bleed through into the final combination.

.. _lsst.cp.pipe.cpDarkTask.CpDarkTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpDarkTask.CpDarkTask

.. _lsst.cp.pipe.cpDarkTask.CpDarkTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpDarkTask.CpDarkTask

.. _lsst.cp.pipe.cpDarkTask.CpDarkTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpDarkTask.CpDarkTask
