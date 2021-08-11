.. lsst-task-topic:: lsst.cp.pipe.cpSkyTask.CpSkyImage

##############
CpSkyImageTask
##############

``CpSkyImageTask`` preprocesses exposures after :lsst-task:`~lsst.ip.isr.IsrTask` to mask detections so a cleaner background estimate can be measured.

.. _lsst.cp.pipe.cpSkyTask.CpSkyImage-processing-summary:

Processing summary
==================

``CpSkyImageTask`` runs these operations:

#. Run :lsst-task:`~lsst.pipe.drivers.background.MaskObjectsTask` to identify and mask sources in the image.
#. Construct a single-detector `~lsst.pipe.drivers.background.FocalPlaneBackground` model from the detection clean image.

.. _lsst.cp.pipe.cpSkyTask.CpSkyImage-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpSkyTask.CpSkyImage

.. _lsst.cp.pipe.cpSkyTask.CpSkyImage-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpSkyTask.CpSkyImage

.. _lsst.cp.pipe.cpSkyTask.CpSkyImage-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpSkyTask.CpSkyImage
