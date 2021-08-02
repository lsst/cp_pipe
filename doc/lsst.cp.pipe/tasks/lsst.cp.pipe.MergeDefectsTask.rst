 
.. lsst-task-topic:: lsst.cp.pipe.MergeDefectsTask

################
MergeDefectsTask
################

``MergeDefectsTask`` combines all of the partial defect sets from the individual exposure measurements into a complete final defect set.

.. _lsst.cp.pipe.MergeDefectsTask-processing-summary:

Processing summary
==================

``MergeDefectsTask`` runs these operations:

#. Combine all input partial defect sets by the input image type (usually dark and flat exposures) based on the fraction of inputs that have a defect in each pixel.
#. Create the final defect set from the union of all the per-image type defects.
#. Optionally mask the edges of the detectors.

.. _lsst.cp.pipe.MergeDefectsTask-api:

Python API summary
==================


.. _lsst.cp.pipe.MergeDefectsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.MergeDefectsTask

.. _lsst.cp.pipe.MergeDefectsTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.MergeDefectsTask

.. _lsst.cp.pipe.MergeDefectsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.MergeDefectsTask
