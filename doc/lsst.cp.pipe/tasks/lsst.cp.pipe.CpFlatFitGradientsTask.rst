.. lsst-task-topic:: lsst.cp.pipe.CpFlatFitGradientsTask

######################
CpFlatFitGradientsTask
######################

``CpFlatFitGradientsTask`` is a task to fit large scale radial and planar gradients in the full field-of-view, with emphasis on LSSTCam flat-fields.
This task is intended to be used in two ways:

#. The task can be applied to a sky flat to generate a reference gradient calibration product, with the default name of ``flat_gradient_reference`` (with one per physical filter).
#. The task can be applied to a dome flat to generate an intermediate gradient product, with the default name of ``flat_gradient``.

There is a second task, :doc:`lsst.cp.pipe.CpFlatApplyGradientsTask`, that can be used to remove any planar gradients from a dome flat and adjust the radial function so that it matches the reference gradient generated from the sky flat.

Processing Summary
==================

``CpFlatFitGradientsTask`` runs this sequence of operations:

#. All flats over the full focal plane are loaded, defects are applied, they are rebinned according to the configuration, and the bins are projected into focal plane coordinates (with x and y in millimeters).
#. The flat may be normalized over the central region (rather than the default focal-plane average).
#. A spline model to the radial gradient is fit, along with a nuisance parameter for the average relative throughput of the ITL to E2V detectors.
   Additionally, a planar gradient may be fit; an extra planar gradient that only applies at large radii; and an offset to the centroid for the radial spline model.
#. Basic QA plots showing the focal plane model and residuals, along with the radial function, are created.
#. The gradient is output, with the precise dataset type determined by whether this is a reference gradient or a dome gradient.

.. _lsst.cp.pipe.CpFlatFitGradientsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CpFlatFitGradientsTask

.. _lsst.cp.pipe.CpFlatFitGradientsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CpFlatFitGradientsTask
