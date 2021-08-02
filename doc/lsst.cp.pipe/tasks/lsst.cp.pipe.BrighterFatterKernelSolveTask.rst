.. lsst-task-topic:: lsst.cp.pipe.BrighterFatterKernelSolveTask

#############################
BrighterFatterKernelSolveTask
#############################

``BrighterFatterKernelSolveTask`` inverts the covariance matrix from a photon transfer curve dataset to produce a brighter-fatter kernel.

.. _lsst.cp.pipe.BrighterFatterKernelSolveTask-processing-summary:

Processing summary
==================

``BrighterFatterKernelSolveTask`` runs these operations:

#. Scale and normalize the covariance matrix.
#. Tile the covariance matrix to produce the cross-correlation in all four quadrants.
#. Invert cross-correlation through successive over relaxation process.
#. Optionally average the per-amplifier kernels into a per-detector kernel.

.. _lsst.cp.pipe.BrighterFatterKernelSolveTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.BrighterFatterKernelSolveTask

.. _lsst.cp.pipe.BrighterFatterKernelSolveTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.BrighterFatterKernelSolveTask

.. _lsst.cp.pipe.BrighterFatterKernelSolveTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.BrighterFatterKernelSolveTask
