.. lsst-task-topic:: lsst.cp.pipe.CalibCombineTask

################
CalibCombineTask
################

``CalibCombineTask`` scales and coadds the input processed calibration exposures to produce a final master calibration.

.. _lsst.cp.pipe.CalibCombineTask-processing-summary:

Processing summary
==================

``CalibCombineTask`` runs these operations:

#. Determine the scale factors to apply to each input exposure.
#. Apply the scaling to the input exposures.
#. Combine the inputs using `~lsst.afw.math.statisticsStack`.
#. Interpolate NaN pixels.
#. Optionally mask the vignetted region.

.. _lsst.cp.pipe.CalibCombineTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CalibCombineTask

.. _lsst.cp.pipe.CalibCombineTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.CalibCombineTask

.. _lsst.cp.pipe.CalibCombineTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CalibCombineTask
