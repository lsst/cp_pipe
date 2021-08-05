.. lsst-task-topic:: lsst.cp.pipe.MeasureDefectsTask

####################
MeasureDefectsTask
####################

.. _lsst.cp.pipe.MeasureDefectsTask-processing-summary:

Processing summary
==================

``MeasureDefectsTask`` runs these operations in a loop over amplifiers:

#. Measures the clipped mean to estimate the background level.
#. Subtracts that background from the image.
#. Identifies pixels that are either above or below the configured sigma thresholds.
#. Constructs a `~lsst.ip.isr.Defects` object from the footprint set of those pixels.

.. _lsst.cp.pipe.MeasureDefectsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.MeasureDefectsTask

.. _lsst.cp.pipe.MeasureDefectsTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.MeasureDefectsTask

.. _lsst.cp.pipe.MeasureDefectsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.MeasureDefectsTask

.. _lsst.cp_pipe.MeasureDefectsTask-debug:

Debugging
=========

ampFlux
    Display a histogram of the pixel distribution for the amplifier (`bool`)?

historgram
    Display the amplifier with the defects overplotted (`bool`)?
