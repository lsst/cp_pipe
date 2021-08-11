.. lsst-task-topic:: lsst.cp.pipe.CrosstalkExtractTask

####################
CrosstalkExtractTask
####################

``CrosstalkExtractTask`` measures the flux ratios between bright sources on one amplifier and the same location on the other amplifiers to look for crosstalk sources.

.. _lsst.cp.pipe.CrosstalkExtractTask-processing-summary:

Processing summary
==================

``CrosstalkExtractTask`` runs these operations:

#. Identifies bright sources on each amplifier.
#. Iterates over the other amplifiers, extracting those potential targets to ensure they have the readout corner placed at the same amplifier coordinate.
#. Measures the ratio between the background subtracted target location and the source amplifier location.

.. _lsst.cp.pipe.CrosstalkExtractTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CrosstalkExtractTask

.. _lsst.cp.pipe.CrosstalkExtractTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.CrosstalkExtractTask

.. _lsst.cp.pipe.CrosstalkExtractTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CrosstalkExtractTask

.. _lsst.cp_pipe.CrosstalkExtractTask-debug:

Debugging
=========

extract
    Display the exposure under consideration, with the pixels used for crosstalk measurement indicated by the DETECTED mask plane (`bool`)?

pixels
    Display a plot of the ratio calculated for each pixel used in this exposure, split by amplifier pairs.  The median value is listed for reference (`bool`)?
