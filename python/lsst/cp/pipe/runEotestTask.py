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

"""Calibration products production task code."""
from __future__ import absolute_import, division, print_function

import os
import glob
import sys

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.log as lsstLog
import lsst.eotest.sensor as sensorTest


class RunEotestConfig(pexConfig.Config):
    """Config class for the calibration products production (CP) task."""

    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'",
        default='ccd',
    )
    fe55 = pexConfig.ConfigurableField(
        target=sensorTest.Fe55Task,
        doc="The Fe55 analysis task.",
    )
    doFe55 = pexConfig.Field(
        dtype=bool,
        doc="Measure gains using Fe55?",
        default=True,
    )
    readNoise = pexConfig.ConfigurableField(
        target=sensorTest.ReadNoiseTask,
        doc="The read noise task.",
    )
    doReadNoise = pexConfig.Field(
        dtype=bool,
        doc="Measure the read-noise?",
        default=True,
    )
    brightPixels = pexConfig.ConfigurableField(
        target=sensorTest.BrightPixelsTask,
        doc="The bright pixel/column finding task.",
    )
    doBrightPixels = pexConfig.Field(
        dtype=bool,
        doc="Find bright pixels?",
        default=True,
    )
    darkPixels = pexConfig.ConfigurableField(
        target=sensorTest.DarkPixelsTask,
        doc="The dark pixel/column finding task.",
    )
    doDarkPixels = pexConfig.Field(
        dtype=bool,
        doc="Find dark pixels?",
        default=True,
    )
    traps = pexConfig.ConfigurableField(
        target=sensorTest.TrapTask,
        doc="The trap-finding task.",
    )
    doTraps = pexConfig.Field(
        dtype=bool,
        doc="Find traps using pocket-pumping exposures?",
        default=True,
    )
    cte = pexConfig.ConfigurableField(
        target=sensorTest.CteTask,
        doc="The CTE analysis task.",
    )
    doCTE = pexConfig.Field(
        dtype=bool,
        doc="Measure the charge transfer efficiency?",
        default=True,
    )
    ptc = pexConfig.ConfigurableField(
        target=sensorTest.PtcTask,
        doc="The PTC analysis task.",
    )
    doPTC = pexConfig.Field(
        dtype=bool,
        doc="Measure the photon transfer curve?",
        default=True,
    )
    flatPair = pexConfig.ConfigurableField(
        target=sensorTest.FlatPairTask,
        doc="The flat-pair analysis task.",
    )
    doFlatPair = pexConfig.Field(
        dtype=bool,
        doc="Measure the detector response vs incident flux using flat pairs?",
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
        # TODO: Set to proper values - DM-12939
        self.fe55.temp_set_point = -100
        self.fe55.temp_set_point_tol = 20

        # TODO: Set to proper values - DM-12939
        self.readNoise.temp_set_point = -100
        self.readNoise.temp_set_point_tol = 20

        # TODO: make this work - DM-12939
        self.brightPixels.temp_set_point = -100
        self.brightPixels.temp_set_point_tol = 20

        # TODO: find the proper settings for flatPairTask to work. This will mean either using the expTime,
        # or working out what's wrong with the MONDIODE values (debug that anyway as a large difference might
        # indicate something else going on). Note that this task doesn't really use much in its config class,
        # but passes in things at runtime param to its run() method, hence putting in a slot for them here.
        # DM-12939
        self.flatPairMaxPdFracDev = 0.99

    def validate(self):
        """Override of the valiate() method.

        The pexConfigs of the subTasks here cannot be validated in the normal way, as they are the configs
        for eotest, which does illegal things, and this would require an upstream PR to fix. Therefore, we
        override the validate() method here, and use it to set the output directory for each of the tasks
        based on the legal pexConfig parameter for the main task.
        """
        log = lsstLog.Log.getLogger("cp.pipe.runEotestConfig")
        if not self.eotestOutputPath:
            raise RuntimeError("Must supply an output path for eotest data. "
                               "Please set config.eotestOutputPath.")

        taskList = ['fe55', 'brightPixels', 'darkPixels', 'readNoise', 'traps', 'cte', 'flatPair', 'ptc']
        for task in taskList:
            if getattr(self, task).output_dir != '.':
                # Being thorough here: '.' is the eotest default. If this is not the value then the user has
                # specified something, and we're going to clobber it, so raise a warning. Unlike to happen.
                log.warn("OVERWRITING: Found a user defined output path of %s for %sTask. "
                         "This has been overwritten with %s, as individually specified output paths for "
                         "subTasks are not supported at present" % (getattr(self, task).output_dir,
                                                                    task, self.eotestOutputPath))
            getattr(self, task).output_dir = self.eotestOutputPath


class RunEotestTask(pipeBase.CmdLineTask):
    """
    Task to run test stand data through eotest using a butler.

    This task is used to produce an eotest report (the project's sensor
    acceptance testing package)
    Examples of some of its operations are as follows:
    * Given a set of flat-field images, find the dark pixels and columns.
    * Given a set of darks, find the bright pixels and columns.
    * Given a set of Fe55 exposures, calulate the gain of the readout chain,
        in e-/ADU
    * Given a set of Fe55 exposures, calulate the instrinsic PSF of the silicon,
        and the degradation of
    * the PSF due to CTE.
    * Given a set of flat-pairs, measure the photon transfer curve (PTC).
    * Given a set of bias frames, calculate the read noise of the system in e-.
    * Given a set of pocket-pumping exposures, find charge-traps in the silicon.

    The RunEotestTask.runEotestDirect() is only applicable to LSST sensors, and
    only for a specific type of dataset. This method takes a
    dafPersistance.Butler corresponding to a repository in which a full eotest
    run has been taken and ingested, and runs each of the tasks in eotest
    directly, allowing for bitwise comparison with results given by the camera
    team.

    See http://ls.st/ldm-151 Chapter 4, Calibration Products Production for
    further details regarding the inputs and outputs.
    """

    ConfigClass = RunEotestConfig
    _DefaultName = "runEotest"

    def __init__(self, *args, **kwargs):
        """Constructor for the RunEotestTask."""
        if 'lsst.eotest.sensor' not in sys.modules:  # check we have eotest before going further
            raise RuntimeError('eotest failed to import')

        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)

        # Note - we can't currently call validate on the subTask configs, as they are NOT valid
        # due to state of eotest. However, we override validate() and call it here
        # and use it to set the output dir config parameter in the subTasks.
        self.config.validate()
        self.config.freeze()

        self.makeSubtask("fe55")
        self.makeSubtask("readNoise")
        self.makeSubtask("brightPixels")
        self.makeSubtask("darkPixels")
        self.makeSubtask("traps")
        self.makeSubtask("cte")
        self.makeSubtask("flatPair")
        self.makeSubtask("ptc")

    def _getMaskFiles(self, path, ccd):
        """Get all available eotest mask files for a given ccd.

        Each stage of the processing generates more mask files, so this allows each to be picked up
        as more and more tests run, and saves having to have clever logic for if some tasks fail.

        Parameters
        ----------
        path : `str`
            Path on which to find the mask files
        ccd : `string` or `int`
            Name/identifier of the CCD

        Returns
        -------
        maskFiles : iterable of `str`
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
        path : `str`
           Path on which to delete all the eotest medianed files.
        """
        for filename in glob.glob(os.path.join(path, '*_median_*.fits')):
            os.remove(filename)

    def makeEotestReport(self, butler):
        """After running eotest, generate pdf(s) of the results.

        Generate a sensor test report from the output data in config.eotestOutputPath, one for each CCD.
        The pdf file(s), along with the .tex file(s) and the individual plots are written
        to the eotestOutputPath.
        .pdf generation requires a TeX distro including pdflatex to be installed.
        """
        ccds = butler.queryMetadata('raw', self.config.ccdKey)
        for ccd in ccds:
            self.log.info("Starting test report generation for %s"%ccd)
            try:
                plotPath = os.path.join(self.config.eotestOutputPath, 'plots')
                if not os.path.exists(plotPath):
                    os.makedirs(plotPath)
                plots = sensorTest.EOTestPlots(ccd, self.config.eotestOutputPath, plotPath)
                eoTestReport = sensorTest.EOTestReport(plots, wl_dir='')
                eoTestReport.make_figures()
                eoTestReport.make_pdf()
            except Exception as e:
                self.log.warn("Failed to make eotest report for %s: %s"%(ccd, e))
        self.log.info("Finished test report generation.")

    @pipeBase.timeMethod
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
        http://lsst-camera.slac.stanford.edu/eTraveler/exp/LSST-CAMERA/displayProcess.jsp?processPath=1179

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
        * linearityTask()
        * fe55CteTask()
        * eperTask()
        * crosstalkTask()
        * persistenceTask()

        # TODO: For each eotest task, find out what the standard raft testing does for the optional params.
        i.e. many have optional params for gains, bias-frames etc - if we want bitwise identicallity then we
        need to know what is typically provided to these tasks when the camera team runs this code.
        This can probably be worked out from https://github.com/lsst-camera-dh/lcatr-harness
        but it sounds like Jim Chiang doesn't recommend trying to do that.
        DM-12939

        Parameters
        ----------
        butler : `lsst.daf.persistence.butler`
            Butler for the repo containg the eotest data to be used
        run : `str` or `int`
            Optional run number, to be used for repos containing multiple runs
        """
        self.log.info("Running eotest routines direct")

        # Input testing to check that run is in the repo
        runs = butler.queryMetadata('raw', ['run'])
        if run is None:
            if len(runs) == 1:
                run = runs[0]
            else:
                raise RuntimeError("Butler query found %s for runs. eotest datasets must have a run number,"
                                   "and you must specify which run to use if a respoitory contains several."
                                   % runs)
        else:
            run = str(run)
            if run not in runs:
                raise RuntimeError("Butler query found %s for runs, but the run specified (%s) "
                                   "was not among them." % (runs, run))
        del runs  # we have run defined now, so remove this to avoid potential confusion later

        if not os.path.exists(self.config.eotestOutputPath):
            os.makedirs(self.config.eotestOutputPath)

        ccds = butler.queryMetadata('raw', self.config.ccdKey)
        imTypes = butler.queryMetadata('raw', ['imageType'])
        testTypes = butler.queryMetadata('raw', ['testType'])

        ################################
        ################################
        # Run the Fe55 task
        if self.config.doFe55:
            fe55TaskDataId = {'run': run, 'testType': 'FE55', 'imageType': 'FE55'}
            self.log.info("Starting Fe55 pixel task")
            for ccd in ccds:
                if 'FE55' not in testTypes:
                    msg = "No Fe55 tests found. Available data: %s" % testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping Fe55 task")
                        break
                fe55Filenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                    self.config.ccdKey: ccd})[0][:-3]
                                 for visit in butler.queryMetadata('raw', ['visit'], dataId=fe55TaskDataId)]
                self.log.trace("Fe55Task: Processing %s with %s files" % (ccd, len(fe55Filenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = self.fe55.run(sensor_id=ccd, infiles=fe55Filenames, mask_files=maskFiles)
                # gainsPropSet = dafBase.PropertySet()
                # for amp, gain in gains.items():  # there is no propSet.fromDict() method so make like this
                #     gainsPropSet.addDouble(str(amp), gain)
                butler.put(gains, 'eotest_gain', dataId={self.config.ccdKey: ccd, 'run': run})
            del fe55TaskDataId

        # TODO: validate the results above, and/or change code to (be able to) always run
        # over all files instead of stopping at the "required accuracy"
        # This will require making changes to the eotest code.
        # DM-12939

        ################################
        ################################
        # Run the Noise task
        if self.config.doReadNoise:
            # note that LCA-10103 defines the Fe55 bias frames as the ones to use here
            self.log.info("Starting readNoise task")
            noiseTaskDataId = {'run': run, 'testType': 'FE55', 'imageType': 'BIAS'}
            for ccd in ccds:
                if ('FE55' not in testTypes) or ('BIAS' not in imTypes):
                    msg = "Required data for readNoise unavailable. Available data:\
                           \ntestTypes: %s\nimageTypes: %s" % (testTypes, imTypes)
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping noise task")
                noiseFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                     self.config.ccdKey: ccd})[0][:-3]
                                  for visit in butler.queryMetadata('raw', ['visit'],
                                                                    dataId=noiseTaskDataId)]
                self.log.trace("Fe55Task: Processing %s with %s files" % (ccd, len(noiseFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = butler.get('eotest_gain', dataId={self.config.ccdKey: ccd, 'run': run})
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
                if 'DARK' not in testTypes:
                    msg = "No dark tests found. Available data: %s" % testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping bright pixel task")
                        break
                darkFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                    self.config.ccdKey: ccd})[0][:-3]
                                 for visit in butler.queryMetadata('raw', ['visit'],
                                                                   dataId=brightTaskDataId)]
                self.log.trace("BrightTask: Processing %s with %s files" % (ccd, len(darkFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = butler.get('eotest_gain', dataId={self.config.ccdKey: ccd, 'run': run})
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
                    msg = "No superflats found. Available data: %s" % testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping dark pixel task")
                        break
                sflatFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                     self.config.ccdKey: ccd})[0][:-3]
                                  for visit in butler.queryMetadata('raw', ['visit'],
                                                                    dataId=darkTaskDataId)]
                self.log.trace("DarkTask: Processing %s with %s files" % (ccd, len(sflatFilenames)))
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
                    msg = "No pocket pumping exposures found. Available data: %s" % testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping trap task")
                        break
                trapFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                    self.config.ccdKey: ccd})[0][:-3]
                                 for visit in butler.queryMetadata('raw', ['visit'], dataId=trapTaskDataId)]
                if len(trapFilenames) != 1:  # eotest can't handle more than one
                    msg = "Trap Task: Found more than one ppump trap file: %s" % trapFilenames
                    msg += " Running using only the first one found."
                    self.log.warn(msg)
                self.log.trace("Trap Task: Processing %s with %s files" % (ccd, len(trapFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = butler.get('eotest_gain', dataId={self.config.ccdKey: ccd, 'run': run})
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
                    msg = "No superflats found. Available data: %s" % testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping CTE task")
                        break
                sflatFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                     self.config.ccdKey: ccd})[0][:-3]
                                  for visit in butler.queryMetadata('raw', ['visit'], dataId=cteTaskDataId)]
                self.log.trace("CTETask: Processing %s with %s files" % (ccd, len(sflatFilenames)))
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
                    msg = "No dataset for flat_pairs found. Available data: %s" % testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping flatPair task")
                        break
                flatPairFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                self.config.ccdKey: ccd})[0][:-3]
                                     for visit in butler.queryMetadata('raw', ['visit'],
                                                                       dataId=flatPairDataId)]
                # Note that eotest needs the original filename as written by the test-stand data acquisition
                # system, as that is the only place the flat pair-number is recorded, so we have to resolve
                # sym-links and pass in the *original* paths/filenames here :(
                # Also, there is no "flat-pair" test type, so all FLAT/FLAT imType/testType will appear here
                # so we need to filter these for only the pair acquisitions (as the eotest code looks like it
                # isn't totally thorough on rejecting the wrong types of data here)
                # TODO: adding a translator to obs_comCam and ingesting this would allow this to be done
                # by the butler instead of here. DM-12939
                flatPairFilenames = [os.path.realpath(f) for f in flatPairFilenames if
                                     os.path.realpath(f).find('flat1') != -1 or
                                     os.path.realpath(f).find('flat2') != -1]
                if not flatPairFilenames:
                    raise RuntimeError("No flatPair files found.")
                self.log.trace("FlatPairTask: Processing %s with %s files" % (ccd, len(flatPairFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = butler.get('eotest_gain', dataId={self.config.ccdKey: ccd, 'run': run})
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
                    msg = "No dataset for flat_pairs found. Available data: %s" % testTypes
                    if self.config.requireAllEOTests:
                        raise RuntimeError(msg)
                    else:
                        self.log.warn(msg + "\nSkipping PTC task")
                        break
                ptcFilenames = [butler.get('raw_filename', dataId={'visit': visit,
                                                                   self.config.ccdKey: ccd})[0][:-3]
                                for visit in butler.queryMetadata('raw', ['visit'], dataId=ptcDataId)]
                # Note that eotest needs the original filename as written by the test-stand data acquisition
                # system, as that is the only place the flat pair-number is recorded, so we have to resolve
                # sym-links and pass in the *original* paths/filenames here :(
                # Also, there is no "flat-pair" test type, so all FLAT/FLAT imType/testType will appear here
                # so we need to filter these for only the pair acquisitions (as the eotest code looks like it
                # isn't totally thorough on rejecting the wrong types of data here)
                # TODO: adding a translator to obs_comCam and ingesting this would allow this to be done
                # by the butler instead of here. DM-12939
                ptcFilenames = [os.path.realpath(f) for f in ptcFilenames if
                                os.path.realpath(f).find('flat1') != -1 or
                                os.path.realpath(f).find('flat2') != -1]
                if not ptcFilenames:
                    raise RuntimeError("No flatPair files found")
                self.log.trace("PTCTask: Processing %s with %s files" % (ccd, len(ptcFilenames)))
                maskFiles = self._getMaskFiles(self.config.eotestOutputPath, ccd)
                gains = butler.get('eotest_gain', dataId={self.config.ccdKey: ccd, 'run': run})
                self.ptc.run(sensor_id=ccd, infiles=ptcFilenames, mask_files=maskFiles, gains=gains)
            del ptcDataId

        self._cleanupEotest(self.config.eotestOutputPath)
        self.log.info("Finished running EOTest")
