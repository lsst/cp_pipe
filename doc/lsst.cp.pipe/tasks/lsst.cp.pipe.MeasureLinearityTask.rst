.. lsst-task-topic:: lsst.cp.pipe.MeasureLinearityTask

####################
MeasureLinearityTask
####################

``MeasureLinearityTask`` is the gen2 wrapper task around the gen3 implementation task (listed below).

.. _lsst.cp.pipe.MeasureLinearityTask-processing-summary:

Processing summary
==================

``MeasureLinearityTask`` runs these operations:

#. Runs :lsst-task:`~lsst.cp.pipe.LinearitySolveTask` to generate a linearity solution (a `~lsst.cp.pipe.linearity.Linearizer`) from the input photon transfer curve.


.. _lsst.cp.pipe.MeasureLinearityTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.MeasureLinearityTask

.. _lsst.cp.pipe.MeasureLinearityTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.MeasureLinearityTask

.. _lsst.cp.pipe.MeasureLinearityTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.MeasureLinearityTask
