.. lsst-task-topic:: lsst.cp.pipe.MeasureCrosstalkTask

####################
MeasureCrosstalkTask
####################

``MeasureCrosstalkTask`` is a gen2 wrapper task that combines the :lsst-task:`~lsst.cp.pipe.CrosstalkExtractTask` and :lsst-task:`~lsst.cp.pipe.CrosstalkSolveTask` tasks to measure the crosstalk.

.. _lsst.cp.pipe.MeasureCrosstalkTask-processing-summary:

Processing summary
==================

``MeasureCrosstalkTask`` runs these operations:

#. Run `~lsst.cp.pipe.CrosstalkExtractTask` on all input exposures.
#. Run `~lsst.cp.pipe.CrosstalkSolveTask` on the combines set of ratio measurements.

.. _lsst.cp.pipe.MeasureCrosstalkTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.MeasureCrosstalkTask

.. _lsst.cp.pipe.MeasureCrosstalkTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.MeasureCrosstalkTask

.. _lsst.cp.pipe.MeasureCrosstalkTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.MeasureCrosstalkTask

.. _lsst.cp_pipe.MeasureCrosstalkTask-debug:

Debugging
=========

extract
    Display the exposure under consideration, with the pixels used for crosstalk measurement indicated by the DETECTED mask plane (`bool`)?
