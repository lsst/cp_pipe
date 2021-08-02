.. lsst-task-topic:: lsst.cp.pipe.ptc.PhotonTransferCurveSolveTask

############################
PhotonTransferCurveSolveTask
############################

``PhotonTransferCurveSolveTask`` combines the partial photon transfer curve (PTC) datasets from pairs of flats, and produces the complete curve, fitting the gains, read noises, and measuring the covariances.

.. _lsst.cp.pipe.ptc.PhotonTransferCurveSolveTask-processing-summary:

Processing summary
==================

``PhotonTransferCurveSolveTask`` runs these operations:

#. Collates the multiple single-pair input PTC datasets.
#. Fits either a full-covariance model, or one of two simpler approximations (polynomial and Astier+19 exponential approximation).
#. Persists the final complete dataset to disk.

.. _lsst.cp.pipe.ptc.PhotonTransferCurveSolveTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.ptc.PhotonTransferCurveSolveTask

.. _lsst.cp.pipe.ptc.PhotonTransferCurveSolveTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.ptc.PhotonTransferCurveSolveTask

.. _lsst.cp.pipe.ptc.PhotonTransferCurveSolveTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.ptc.PhotonTransferCurveSolveTask
