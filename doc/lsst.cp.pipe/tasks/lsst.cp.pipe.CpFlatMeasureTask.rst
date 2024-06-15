.. lsst-task-topic:: lsst.cp.pipe.CpFlatMeasureTask

#################
CpFlatMeasureTask
#################

``CpFlatMeasureTask`` measures image statistics from the input flat field frames to supply the information needed for :lsst-task:`~lsst.cp.pipe.CpFlatNormalizationTask` to determine approprate scale factors.

.. _lsst.cp.pipe.CpFlatMeasureTask-processing-summary:

Processing summary
==================

``CpFlatMeasureTask`` runs these operations:

#. Optionally masks the vignetted region.
#. Measures detector level clipped mean, clipped sigma, and number of pixels.
#. Measures the same statistics at the amplifier level.

.. _lsst.cp.pipe.CpFlatMeasureTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CpFlatMeasureTask

.. _lsst.cp.pipe.CpFlatMeasureTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.CpFlatMeasureTask

.. _lsst.cp.pipe.CpFlatMeasureTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CpFlatMeasureTask
