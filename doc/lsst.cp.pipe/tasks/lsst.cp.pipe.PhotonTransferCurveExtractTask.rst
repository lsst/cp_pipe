.. lsst-task-topic:: lsst.cp.pipe.ptc.PhotonTransferCurveExtractTask

##############################
PhotonTransferCurveExtractTask
##############################

``PhotonTransferCurveExtractTask`` constructs a photon transfer curve (PTC) dataset from a single pair of flat field exposures.

.. _lsst.cp.pipe.ptc.PhotonTransferCurveExtractTask-processing-summary:

Processing summary
==================

``PhotonTransferCurveExtractTask`` runs these operations:

#. Pairs exposures together by either exposure time (the default) or by exposure id (in the case where the exposure time does not track with the observed flux).
#. Measures the mean, variances, and covariances from the difference of the pair of exposures.
#. Persists these results in a set of partial PTC datasets.

.. _lsst.cp.pipe.ptc.PhotonTransferCurveExtractTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.ptc.PhotonTransferCurveExtractTask

.. _lsst.cp.pipe.ptc.PhotonTransferCurveExtractTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.ptc.PhotonTransferCurveExtractTask

.. _lsst.cp.pipe.ptc.PhotonTransferCurveExtractTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.ptc.PhotonTransferCurveExtractTask
