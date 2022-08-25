.. lsst-task-topic:: lsst.cp.pipe.cpSkyTask.CpSkyImageTask

##############
CpSkyImageTask
##############

``CpSkyImageTask`` preprocesses exposures after :lsst-task:`~lsst.ip.isr.IsrTask` to mask detections so a cleaner background estimate can be measured.

.. _lsst.cp.pipe.cpSkyTask.CpSkyImageTask-processing-summary:

Processing summary
==================

``CpSkyImageTask`` runs these operations:

#. Run :lsst-task:`~lsst.pipe.tasks.background.MaskObjectsTask` to identify and mask sources in the image.
#. Construct a single-detector `~lsst.pipe.tasks.background.FocalPlaneBackground` model from the detection clean image.

.. _lsst.cp.pipe.cpSkyTask.CpSkyImageTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpSkyTask.CpSkyImageTask

.. _lsst.cp.pipe.cpSkyTask.CpSkyImageTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpSkyTask.CpSkyImageTask

.. _lsst.cp.pipe.cpSkyTask.CpSkyImageTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpSkyTask.CpSkyImageTask
