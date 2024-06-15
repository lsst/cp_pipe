.. lsst-task-topic:: lsst.cp.pipe.CpDarkTask

##########
CpDarkTask
##########

``CpDarkTask`` preprocesses exposures after :lsst-task:`~lsst.ip.isr.IsrTask` and before the final dark combination.

.. _lsst.cp.pipe.CpDarkTask-processing-summary:

Processing summary
==================

``CpDarkTask`` runs these operations:

#. Identifies and masks cosmic rays.
#. Optionally grows the cosmic ray masks to ensure they do not bleed through into the final combination.

.. _lsst.cp.pipe.CpDarkTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CpDarkTask

.. _lsst.cp.pipe.CpDarkTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.CpDarkTask

.. _lsst.cp.pipe.CpDarkTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CpDarkTask
