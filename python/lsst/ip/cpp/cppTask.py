#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

# import math
# import numpy
"""Calibration products production task code."""
from __future__ import absolute_import, division, print_function

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.log as lsstLog
# despite living inside the lsst namespace this is not a standard lsst package and usually won't be present
try:
    import lsst.eotest.sensor as sensorTest
except ImportError:
    raise RuntimeError("Error importing eotest")

import os
import glob


class CppTaskConfig(pexConfig.Config):
    """Config class for the calibration products production (CPP) task."""

    fe55 = pexConfig.ConfigurableField(
        target=sensorTest.Fe55Task,
        doc="The eotest Fe55 analysis task. Should not be retragetted.",
    )
    doFe55 = pexConfig.Field(
        dtype=bool,
        doc="Run the Fe55 task to measure gains?",
        default=True,
    )
    readNoise = pexConfig.ConfigurableField(
        target=sensorTest.ReadNoiseTask,
        doc="The eotest read noise task. Should not be retragetted.",
    )
    doReadNoise = pexConfig.Field(
        dtype=bool,
        doc="Run the readNoiseTask to measure read noise?",
        default=True,
    )
    brightPixels = pexConfig.ConfigurableField(
        target=sensorTest.BrightPixelsTask,
        doc="The eotest bright pixel/column finding task. Should not be retragetted.",
    )
    doBrightPixels = pexConfig.Field(
        dtype=bool,
        doc="Run the brightPixelTask to find the bright pixels?",
        default=True,
    )
    darkPixels = pexConfig.ConfigurableField(
        target=sensorTest.DarkPixelsTask,
        doc="The eotest dark pixel/column finding task. Should not be retragetted.",
    )
    doDarkPixels = pexConfig.Field(
        dtype=bool,
        doc="Run the darkPixelsTask to find the dark pixels?",
        default=True,
    )
    traps = pexConfig.ConfigurableField(
        target=sensorTest.TrapTask,
        doc="The eotest trap-finding task. Should not be retragetted.",
    )
    doTraps = pexConfig.Field(
        dtype=bool,
        doc="Run the trapTask to find the traps?",
        default=True,
    )
    cte = pexConfig.ConfigurableField(
        target=sensorTest.CteTask,
        doc="The eotest CTE analysis task. Should not be retragetted.",
    )
    doCTE = pexConfig.Field(
        dtype=bool,
        doc="Run the CTE task to measure the CTE?",
        default=True,
    )
    ptc = pexConfig.ConfigurableField(
        target=sensorTest.PtcTask,
        doc="The eotest PTC analysis task. Should not be retragetted.",
    )
    doPTC = pexConfig.Field(
        dtype=bool,
        doc="Run the PTC task to measure the photon transfer curve?",
        default=True,
    )
    flatPair = pexConfig.ConfigurableField(
        target=sensorTest.FlatPairTask,
        doc="The eotest flat-pair analysis task. Should not be retragetted.",
    )
    doFlatPair = pexConfig.Field(
        dtype=bool,
        doc="Run the flatPair task?",
        default=True,
    )
    eotestOutputPath = pexConfig.Field(
        dtype=str,
        doc="Path to which to write the eotest output results. Madatory runtime arg for running eotest.",
        default='',
    )
    requireAllEOTests = pexConfig.Field(
        dtype=bool,
        doc="If True, all tests are required to be runnable, and will Raise if data is missing. If False, "
        "processing will continue if a previous part failed due to the input dataset being incomplete.",
        default=True,
    )
    flatPairMaxPdFracDev = pexConfig.Field(
        dtype=float,
        doc="Maximum allowed fractional deviation between photodiode currents for the eotest flatPair task. "
        "This value is passed to the task's run() method at runtime rather than being stored in the task's"
        "own pexConfig field.",
        default=0.05,
    )

    def setDefaults(self):
        """Set default config options for the subTasks."""
        # TODO: Set to proper values
        self.fe55.temp_set_point = -100
        self.fe55.temp_set_point_tol = 20

        # TODO: Set to proper values
        self.readNoise.temp_set_point = -100
        self.readNoise.temp_set_point_tol = 20

        # TODO: make this work
        self.brightPixels.temp_set_point = -100
        self.brightPixels.temp_set_point_tol = 20

        # TODO: find the proper settings for flatPairTask to work. This will mean either using the expTime,
        # or working out what's wrong with the MONDIODE values (debug that anyway as a large difference might
        # indicate something else going on). Note that this task doesn't really use much in its config class,
        # but passes in things at runtime param to its run() method, hence putting in a slot for them here.
        self.flatPairMaxPdFracDev = 0.99

    def validate(self):
        """Override of the valiate() method.

        The pexConfigs of the subTasks here cannot be validated in the normal way, as they are the configs
        for eotest, which does illegal things, and this would require an upstream PR to fix. Therefore, we
        override the validate() method here, and use it to set the output directory for each of the tasks
        based on the legal pexConfig parameter for the main task.
        """
        log = lsstLog.Log.getLogger("ip.cpp.cppTaskConfig")
        if not self.eotestOutputPath:
            raise RuntimeError("Must supply an output path for eotest data."
                               "Please set config.eotestOutputPath.")

        taskList = ['fe55', 'brightPixels', 'darkPixels', 'readNoise', 'traps', 'cte', 'flatPair', 'ptc']
        for task in taskList:
            if getattr(self, task).output_dir != '.':
                # Being thorough here: '.' is the eotest default. If this is not the value then the user has
                # specified something, and we're going to clobber it, so raise a warning. Unlike to happen.
                log.warn("OVERWRITING: Found a user defined output path of %s for %sTask. "
                         "This has been overwritten with %s, as individually specified output paths for "
                         "subTasks are not supported at present"%(getattr(self, task).output_dir,
                                                                  task, self.eotestOutputPath))
            getattr(self, task).output_dir = self.eotestOutputPath


class CppTask(pipeBase.CmdLineTask):
    """
    Calibration Products Production (CPP) task.

    This task is used to produce the calibration products required to calibrate cameras.
    Examples of such operations are as follows:
    Given a set of flat-field images, find the dark pixels and columns.
    Given a set of darks, find the bright pixels and columns.
    Given a set of Fe55 exposures, calulate the gain of the readout chain, in e-/ADU
    Given a set of Fe55 exposures, calulate the instrinsic PSF of the silicon, and the degradation of
    the PSF due to CTE.
    Given a set of flat-pairs, measure the photon transfer curve (PTC).
    Given a set of bias frames, calculate the read noise of the system in e-.
    Given a set of pocket-pumping exposures, find charge-traps in the silicon.

    The CppTask.runEotestDirect() is only applicable to LSST sensors, and only for a specific type of dataset
    This method takes a dafPersistance.Butler corresponding to a repository in which a full eotest run has
    been taken and ingested, and runs each of the tasks in eotest directly, allowing for bitwise comparison
    with results given by the camera team.

    See http://ls.st/ldm-151 Chapter 4, Calibration Products Production for further details
    regarding the inputs and outputs.
    """

    ConfigClass = CppTaskConfig
    _DefaultName = "cpp"

    def __init__(self, *args, **kwargs):
        """
        Constructor for CppTask.

        Calls the lsst.pipe.base.task.Task.__init__() method, then sets up the
        various subTasks for calibration products production task.

        Parameters
        ----------
        *args :
            a list of positional arguments passed on to the Task constructor
        **kwargs :
            a dictionary of keyword arguments passed on to the Task constructor
        """
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

        # Note - we can't currently call validate on the subTask configs, as they are NOT valid
        # due to state of eotest. However, we override validate() and call it here
        # and use it to set the output dir config parameter in the subTasks.
        self.config.validate()
        self.config.freeze()

        # One of these for each pexConfig.ConfigurableField
        self.makeSubtask("fe55")
        self.makeSubtask("readNoise")
        self.makeSubtask("brightPixels")
        self.makeSubtask("darkPixels")
        self.makeSubtask("traps")
        self.makeSubtask("cte")
        self.makeSubtask("flatPair")
        self.makeSubtask("ptc")

        # Do we want to switch to the better-named logger below or use the one we get from cmdLineTask?
        # e.g. using self.log = lsstLog.Log.getLogger("ip.cpp.cppTask")?

    def _getMaskFiles(self, path, ccd):
        """Get all available eotest mask files for a given ccd.

        Each stage of the processing generates more mask files, so this allows each to be picked up
        as more and more tests run, and saves having to have clever logic for if some tasks fail.
        Parameters
        ----------
        path : string
            Path on which to find the mask files
        ccd : string/int
            Name/identifier of the CCD
        Returns
        -------
        maskFiles : list/tuple
            List of mask files, or an empty tuple if none are found
        """
        pattern = '*' + str(ccd) + '*mask*'  # the cast to str supports obs_auxTel
        maskFiles = glob.glob(os.path.join(path, pattern))
        return maskFiles if len(maskFiles) > 0 else ()  # eotest wants an empty tuple here

    def _cleanupEotest(self, path):
        """Delete all the medianed files left behind after eotest has run.

        Running eotest generates a lot of interim medianed files, so this just cleans them up.
        Parameters
        ----------
        path : string
           Path on which to delete all the eotest medianed files.
        """
        for filename in glob.glob(os.path.join(path, '*median*')):
            os.remove(filename)

    def _gainPropSetToDict(self, pSet):
        """Translator for the persisted gain values.

        eotest wants an {amp: gain} dictionary with integer keys for the amps.
        When we persist dafBase.propertySets these only take strings for keys,
        so the provided .toDict() method won't do, so here we provide an extra
        layer of translation to keep eotest happy.

        Parameters
        ----------
        pSet : daf.base.PropertySet
           PropertySet to be translated to an integer-keyed dictionary
        """
        return {int(amp): gain for amp, gain in pSet.toDict().items()}

    def testMethod(self):  # TODO: XXX remove this whole method, just useful for debugging/dev work
        """TODO: Remove this docstring that only exists to remove flake8 error."""
        self.log.warn("Test warning message")
        print(self.config.requireAllEOTests)

    # @pipeBase.timeMethod #xxx re-include this
    def runEotestDirect(self, butler, run=None):
        """
        Generate calibration products using eotest algorithms.

        Generate all calibration products possible using the vanilla eotest implementation,
        given a butler for a TS8 (raft-test) repo. It can contain multiple runs, but must correspond to
        only a single raft/RTM.

        - Run all eotest tasks possible, using the butler to gather the data
        - Write outputs in eotest format

        In order to replicate the canonical eotest analysis, the tasks should be run in a specific order.
        This is given/defined in the "Steps" section here:
        http://lsst-camera.slac.stanford.edu/eTraveler/exp/LSST-CAMERA/
               displayProcess.jsp?processPath=1179

        But is replicated here for conveniece:
        * 55Fe Analysis
        * CCD Read Noise Analysis
        * Bright Defects Analysis
        * Dark Defects Analysis
        * Traps Finding
        * Dark Current                  X - will not be implemented here
        * Charge Transfer Efficiencies
        * Photo-response analysis       X - will not be implemented here
        * Flat Pairs Analysis
        * Photon Transfer Curve
        * Quantum Efficiency            X - will not be implemented here

        List of tasks that exist in the eotest package but aren't mentioned on the above link:
        --------
        # linearityTask()
        # fe55CteTask()
        # eperTask()
        # crosstalkTask()
        # persistenceTask()

        # TODO: For each eotest task, find out what the standard raft testing does for the optional params.
        i.e. many have optional params for gains, bias-frames etc - if we want bitwise identicallity then we
        need to know what is typically provided to these tasks when the camera team runs this code.
        This can probably be worked out from https://github.com/lsst-camera-dh/lcatr-harness
        but it sounds like Jim Chiang doesn't recommend trying to do that.

        Parameters
        ----------
        butler : daf.persistence.butler
            Butler for the repo containg the eotest data to be used
        run : string or int
            Optional run number, to be used for repos containing multiple runs
        """
        self.log.info("Running eotest routines direct")
        runs = butler.queryMetadata('raw', ['run'])
        if isinstance(run, int):
            run = str(run)
        if len(runs) != 1 and run is None:  # lots found and we don't know which one to choose
            raise RuntimeError("Butler query found %s for runs. eotest datasets must have a run numbers, and"
                               " must specify which run to use if a respoitory contains multiple runs."%runs)
        elif run is not None and run not in runs:  # Could be specifying one of many, or one of one here
            raise RuntimeError("Butler query found %s for runs, but the run specified (%s) "
                               "was not among them."%(runs, run))
        elif run is None:  # we know it's OK now
            run = butler.queryMetadata('raw', ['run'])[0]
        del runs  # we have run defined now, so remove this to avoid potential confusion later

        if not os.path.exists(self.config.eotestOutputPath):
            os.mkdir(self.config.eotestOutputPath)

        ccds = butler.queryMetadata('raw', ['ccd'])
        imTypes = butler.queryMetadata('raw', ['imageType'])
        testTypes = butler.queryMetadata('raw', ['testType'])

        ################################
        ################################
        # Run the Fe55 task
        if self.config.doFe55:
            fe55TaskDataId = {'run': run, 'testType': 'FE55', 'imageType': 'FE55'}
            self.log.info("Starting Fe55 pixel task")
            for ccd in ccds:
                if 'FE55' not in testTypes:  # TODO: Remove not to test functionality
                    msg = "No Fe55 tests found. Available data: %s"%testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping Fe55 task")
                        break
                fe55Filenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                    'ccd': ccd})[0][:-3]
                                 for visit in butler.queryMetadata('raw', ['visit'], dataId=fe55TaskDataId)]
                self.log.trace("Fe55Task: Processing %s with %s files"%(ccd, len(fe55Filenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = self.fe55.run(sensor_id=ccd, infiles=fe55Filenames, mask_files=maskFiles)
                gainsPropSet = dafBase.PropertySet()
                for amp, gain in gains.items():  # there is no propSet.fromDict() method so make like this
                    gainsPropSet.addDouble(str(amp), gain)
                butler.put(gainsPropSet, 'eotest_gain', dataId={'ccd': ccd, 'run': run})
            del fe55TaskDataId

        # TODO: validate the results above, and/or change code to (be able to) always run
        # over all files instead of stopping at the "required accuracy"
        # This will require making changes to the eotest code.

        ################################
        ################################
        # Run the Noise task
        if self.config.doReadNoise:
            # self.readNoise.config.output_dir = '/home/mfl/thisisbad/'
            # note that LCA-10103 defines the Fe55 bias frames as the ones to use here
            self.log.info("Starting readNoise task")
            noiseTaskDataId = {'run': run, 'testType': 'FE55', 'imageType': 'BIAS'}
            for ccd in ccds:
                if ('FE55' not in testTypes) or ('BIAS' not in imTypes):
                    msg = "Required data for readNoise unavailable. Available data:\
                           \ntestTypes: %s\nimageTypes: %s"%(testTypes, imTypes)
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping Fe55 task")
                noiseFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                     'ccd': ccd})[0][:-3]
                                  for visit in butler.queryMetadata('raw', ['visit'],
                                                                    dataId=noiseTaskDataId)]
                self.log.trace("Fe55Task: Processing %s with %s files"%(ccd, len(noiseFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = self._gainPropSetToDict(butler.get('eotest_gain', dataId={'ccd': ccd, 'run': run}))
                self.readNoise.run(sensor_id=ccd, bias_files=noiseFilenames,
                                   gains=gains, mask_files=maskFiles)
            del noiseTaskDataId

        ################################
        ################################
        # Run the bright task
        if self.config.doBrightPixels:
            self.log.info("Starting bright pixel task")
            brightTaskDataId = {'run': run, 'testType': 'DARK', 'imageType': 'DARK'}
            for ccd in ccds:
                if 'DARK' not in testTypes:  # TODO: Remove not to test functionality
                    msg = "No dark tests found. Available data: %s"%testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping bright pixel task")
                        break
                darkFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                    'ccd': ccd})[0][:-3]
                                 for visit in butler.queryMetadata('raw', ['visit'],
                                                                   dataId=brightTaskDataId)]
                self.log.trace("BrigtTask: Processing %s with %s files"%(ccd, len(darkFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = self._gainPropSetToDict(butler.get('eotest_gain', dataId={'ccd': ccd, 'run': run}))
                self.brightPixels.run(sensor_id=ccd, dark_files=darkFilenames,
                                      mask_files=maskFiles, gains=gains)
            del brightTaskDataId

        ################################
        ################################
        # Run the dark task
        if self.config.doDarkPixels:
            self.log.info("Starting dark pixel task")
            darkTaskDataId = {'run': run, 'testType': 'SFLAT_500', 'imageType': 'FLAT'}
            for ccd in ccds:
                if 'SFLAT_500' not in testTypes:
                    msg = "No superflats found. Available data: %s"%testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping dark pixel task")
                        break
                sflatFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                     'ccd': ccd})[0][:-3]
                                  for visit in butler.queryMetadata('raw', ['visit'],
                                                                    dataId=darkTaskDataId)]
                self.log.trace("DarkTask: Processing %s with %s files"%(ccd, len(sflatFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                self.darkPixels.run(sensor_id=ccd, sflat_files=sflatFilenames, mask_files=maskFiles)
            del darkTaskDataId

        ################################
        ################################
        # Run the trap task
        if self.config.doTraps:
            self.log.info("Starting trap task")
            trapTaskDataId = {'run': run, 'testType': 'TRAP', 'imageType': 'PPUMP'}
            for ccd in ccds:
                if ('TRAP' not in testTypes) and ('PPUMP' not in imTypes):
                    msg = "No pocket pumping exposures found. Available data: %s"%testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping trap task")
                        break
                trapFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                    'ccd': ccd})[0][:-3]
                                 for visit in butler.queryMetadata('raw', ['visit'], dataId=trapTaskDataId)]
                if len(trapFilenames) != 1:  # eotest can't handle more than one
                    self.log.fatal("Trap Task: Found more than one ppump trap file: %s"%trapFilenames)
                self.log.trace("Trap Task: Processing %s with %s files"%(ccd, len(trapFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = self._gainPropSetToDict(butler.get('eotest_gain', dataId={'ccd': ccd, 'run': run}))
                self.traps.run(sensor_id=ccd, pocket_pumped_file=trapFilenames[0],
                               mask_files=maskFiles, gains=gains)
            del trapTaskDataId

        ################################
        ################################
        # Run the CTE task
        if self.config.doCTE:
            self.log.info("Starting CTE task")
            cteTaskDataId = {'run': run, 'testType': 'SFLAT_500', 'imageType': 'FLAT'}
            for ccd in ccds:
                if 'SFLAT_500' not in testTypes:
                    msg = "No superflats found. Available data: %s"%testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping CTE task")
                        break
                sflatFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                     'ccd': ccd})[0][:-3]
                                  for visit in butler.queryMetadata('raw', ['visit'], dataId=cteTaskDataId)]
                self.log.trace("CTETask: Processing %s with %s files"%(ccd, len(sflatFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                self.cte.run(sensor_id=ccd, superflat_files=sflatFilenames, mask_files=maskFiles)
            del cteTaskDataId

        ################################
        ################################
        # Run the flatPair task
        if self.config.doFlatPair:
            self.log.info("Starting flatPair task")
            flatPairDataId = {'run': run, 'testType': 'FLAT', 'imageType': 'FLAT'}
            for ccd in ccds:
                if 'FLAT' not in testTypes:
                    msg = "No dataset for flat_pairs found. Available data: %s"%testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping flatPair task")
                        break
                flatPairFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                'ccd': ccd})[0][:-3]
                                     for visit in butler.queryMetadata('raw', ['visit'],
                                                                       dataId=flatPairDataId)]
                # Note that eotest needs the original filenames as that is the only place the flat
                # pair-number is kept, so we have to resolve links and pass in the *original* paths here :(
                # Also, there is no "flat-pair" test type, so all FLAT/FLAT imType/testType will appear here
                # so we need to filter these for only the pair acquisitions (as the eotest code looks like it
                # isn't totally thorough on rejecting the wrong types of data here)
                # TODO: adding a translator to obs_comCam and ingesting this would allow this to be done
                # by the butler instead of here.
                flatPairFilenames = [os.path.realpath(_) for _ in flatPairFilenames if
                                     os.path.realpath(_).find('flat1') != -1 or
                                     os.path.realpath(_).find('flat2') != -1]
                if not flatPairFilenames:
                    raise RuntimeError("No flatPair files found.")
                self.log.trace("FlatPairTask: Processing %s with %s files"%(ccd, len(flatPairFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = self._gainPropSetToDict(butler.get('eotest_gain', dataId={'ccd': ccd, 'run': run}))
                self.flatPair.run(sensor_id=ccd, infiles=flatPairFilenames, mask_files=maskFiles,
                                  gains=gains, max_pd_frac_dev=self.config.flatPairMaxPdFracDev)
            del flatPairDataId

        ################################
        ################################
        # Run the PTC task
        if self.config.doPTC:
            self.log.info("Starting PTC task")
            ptcDataId = {'run': run, 'testType': 'FLAT', 'imageType': 'FLAT'}
            for ccd in ccds:
                if 'FLAT' not in testTypes:
                    msg = "No dataset for flat_pairs found. Available data: %s"%testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping PTC task")
                        break
                ptcFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                   'ccd': ccd})[0][:-3]
                                for visit in butler.queryMetadata('raw', ['visit'], dataId=ptcDataId)]
                # Note that eotest needs the original filenames as that is the only place the flat
                # pair-number is kept, so we have to resolve links and pass in the *original* paths here :(
                # Also, there is no "flat-pair" test type, so all FLAT/FLAT imType/testType will appear here
                # so we need to filter these for only the pair acquisitions (as the eotest code looks like it
                # isn't totally thorough on rejecting the wrong types of data here)
                # TODO: adding a translator to obs_comCam and ingesting this would allow this to be done
                # by the butler instead of here.
                ptcFilenames = [os.path.realpath(_) for _ in ptcFilenames if
                                os.path.realpath(_).find('flat1') != -1 or
                                os.path.realpath(_).find('flat2') != -1]
                if not ptcFilenames:
                    raise RuntimeError("No flatPair files found")
                self.log.trace("PTCTask: Processing %s with %s files"%(ccd, len(ptcFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = self._gainPropSetToDict(butler.get('eotest_gain', dataId={'ccd': ccd, 'run': run}))
                self.ptc.run(sensor_id=ccd, infiles=ptcFilenames, mask_files=maskFiles, gains=gains)
            del ptcDataId

        self._cleanupEotest(self.config.eotestOutputPath)
        self.log.info("Finished running EOTest")
