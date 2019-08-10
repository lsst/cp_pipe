.. lsst-task-topic:: lsst.cp.pipe.FindDefectsMasterTask

#####################
FindDefectsMasterTask
#####################

``FindDefectsMasterTask`` is the `lsst.pipe.base.CmdLineTask` that
is responsible for negotiating with the butler to prepare the
inputs for `FindDefectsTask`;  `FindDefectsTask` does the
actual work of
processes the dark and/or flat images looking for CCD defects
such as hot pixels and blocked columns which it returns to
this task to be persisted.

.. _lsst.cp.pipe.FindDefectsMasterTask-summary:

Processing summary
==================

``FindDefectsMasterTask`` runs this sequence of operations:

- Obtain the input raw ``Exposure`` s from the butler
- Run `lsst.ip.isr.IsrTask` to remove unwanted instrumental
  effects (such as overscan levels)
- Call ``FindDefectsTask.run()`` which returns a `lsst.meas.algorithms.Defect`
- Persists the defects
- Optionally makes plots

.. _lsst.cp.pipe.FindDefectsMasterTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.FindDefectsMasterTask

.. _lsst.cp.pipe.FindDefectsMasterTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.defects.FindDefectsMasterTask

.. _lsst.cp.pipe.FindDefectsMasterTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.defects.FindDefectsMasterTask

.. _lsst.cp.pipe.FindDefectsMasterTask-examples:

Examples
========

The command line utility ``findDefects.py`` actually calls `FindDefectsMasterTask`
and then `FindDefectsTask` to do all the work to generate defects files.

.. code-block:: sh

   findDefects.py /project/shared/comCam --rerun rhl/defects \
                --id detector=189..197 --visitList=201904182050253^201904182051463

A more complex version would be:

.. code-block:: sh

   findDefects.py /project/shared/comCam \
		--log CameraMapper=FATAL LsstCamMapper=FATAL findDefectsMaster.isrForFlats=INFO \
                --rerun rhl/defects \
                --id detector=189..197 --visitList=201904182050253^201904182051463 \
                --config makePlots=True displayBackend=virtualDevice

.. _lsst.cp.pipe.FindDefectsMasterTask-debug:

Debugging
=========

The ``lsst.pipe.base.cmdLineMasterTask.CmdLineTask`` command line task interface supports a flag `-d` to
import `debug.py` from your PYTHONPATH; see ``lsstDebug`` for more about debug.py files.

``FindDefectsMasterTask`` doesn't currently have any debug variables, but if `display` were
supported, you'd enable it with something like:

.. code-block:: py

    import lsstDebug
    def DebugInfo(name):
        debug = lsstDebug.getInfo(name) # N.b. lsstDebug.Info(name) would call us recursively
        if name == "lsst.cp.pipe.defects":
            debug.display = True

        return debug

    lsstDebug.Info = DebugInfo

into your debug.py file and run this task with the --debug flag.
