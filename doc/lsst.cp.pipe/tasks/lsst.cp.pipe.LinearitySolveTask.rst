.. lsst-task-topic:: lsst.cp.pipe.LinearitySolveTask

##################
LinearitySolveTask
##################

``LinearitySolveTask`` constructs a linearity correction model based on the results stored in the input photon transfer curve (PTC) dataset.

.. _lsst.cp.pipe.LinearitySolveTask-processing-summary:

Processing summary
==================

``LinearitySolveTask`` runs these operations:

#. Convert the input exposure time/photodiode flux measurement to a proxy flux by fitting the low-flux end with a linear fit.
#. Perform fit against using the observed flux and this linear proxy flux (using either a spline or a polynomial).
#. Store the correction, such that the corrected flux is equal to the uncorrected flux + the linearity correction as a function of the uncorrected flux.


.. _lsst.cp.pipe.LinearitySolveTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.LinearitySolveTask

.. _lsst.cp.pipe.LinearitySolveTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.LinearitySolveTask

.. _lsst.cp.pipe.LinearitySolveTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.LinearitySolveTask

.. _lsst.cp.pipe.LinearitySolveTask-debug:

Debugging
=========

linearFit
    Display the linearity solution after the initial linear fit (`bool`)?

polyFit
    Display the linearity solution after generating the polynomial fit (`bool`)?

splineFit
    Display the linearity solution after generating the spline fit (`bool`)?

solution
    Display the final linearity solution (`bool`)?
