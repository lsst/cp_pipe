.. lsst-task-topic:: lsst.cp.pipe.FindDefectsTask

###############
FindDefectsTask
###############

``FindDefectsTask`` processes dark and/or flat images looking for CCD defects
such as hot pixels and blocked columns.

.. _lsst.cp.pipe.FindDefectsTask-summary:

Processing summary
==================

.. If the task does not break work down into multiple steps, don't use a list.
.. Instead, summarize the computation itself in a paragraph or two.

``FindDefectsTask`` runs this sequence of operations:

- Call ``lsst.cp.pipe.FindDefectsTask.findHotAndColdPixels`` on each dark and
  flat exposure of a given CCD in turn, looking for pixels that are fainter or brighter than
  would be expected statistically
- Merge all the resulting "dark" and "flat" defects into two sets of defects
- Merge these two sets into one final `lsst.meas.algorithms.Defect`

.. _lsst.cp.pipe.FindDefectsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.FindDefectsTask

.. _lsst.cp.pipe.FindDefectsTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.FindDefectsTask

.. _lsst.cp.pipe.FindDefectsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.FindDefectsTask

.. _lsst.cp.pipe.FindDefectsTask-examples:

Examples
========

Let's start by getting our data from the butler:

.. code-block:: py

    import lsst.daf.persistence as dafPersist

    butler = dafPersist.Butler(os.path.join("/project/shared/comCam", "rerun", "rhl/defects"))
    dataId = dict(detector=194)

    visits = [201904182050253, 201904182051463]

    flatExposures = {}
    for visit in visits:
        flatExposures[visit] = butler.get("postISRCCD", dataId, visit=visit)	

You will, of course, have to run the ISR first!  One way would be to run ``runIsr.py``;
another would be to run the shell command ``findDefects.py`` which, as a byproduct,
leaves the `postISRCCD` files behind.

Now import and initialise the task:

.. code-block:: py

   from lsst.cp.pipe.defects import FindDefectsTask

   config = FindDefectsTask.ConfigClass()
   findDefectsTask = FindDefectsTask(config=config)

We can now actually find our defects:

.. code-block:: py

    ret = findDefectsTask.run({}, exps)
    defects = ret.defects

We can visualise them with e.g.

.. code-block:: py

    import lsst.afw.display as afwDisplay

    visit = visits[1]
    exp = exps[visit].clone()

    exp.mask.addMaskPlane("DEFECT")
    afwDisplay.setDefaultMaskPlaneColor("DEFECT", afwDisplay.YELLOW)

    defects.maskPixels(exp.maskedImage, "DEFECT")

    disp = afwDisplay.Display(1, reopenPlot=True)
    disp.scale('linear', 'zscale')

    disp.mtv(exp, title=f"visit={visit} detector={exp.getDetector().getName()}")
		
.. _lsst.cp.pipe.FindDefectsTask-debug:

Debugging
=========

The ``lsst.pipe.base.cmdLineMasterTask.CmdLineTask`` command line task interface supports a flag `-d` to
import `debug.py` from your PYTHONPATH; see ``lsstDebug`` for more about debug.py files.  This mechanism
may be used to turn on debugging output from any code within the module.

``FindDefectsTask`` doesn't currently have any debug variables, but if `display` were
supported, you'd enable it with something like:

.. code-block:: py

    import lsstDebug
    def DebugInfo(name):
        debug = lsstDebug.getInfo(name) # N.b. lsstDebug.Info(name) would call us recursively
        if name == "lsst.cp.pipe.defects":
            debug.display = True

        return debug

    lsstDebug.Info = DebugInfo

into your debug.py file and run the command line task with the --debug flag.
