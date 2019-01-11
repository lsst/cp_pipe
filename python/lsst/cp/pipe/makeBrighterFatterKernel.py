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
"""Calculation of brighter-fatter effect correlations and kernels."""

__all__ = ['MakeBrighterFatterKernelTaskConfig',
           'MakeBrighterFatterKernelTask',
           'calcBiasCorr']

import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
# the following import is required for 3d projection
from mpl_toolkits.mplot3d import axes3d   # noqa: F401
import pickle as pkl

import lsstDebug
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.display as afwDisp
from lsst.ip.isr import IsrTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


class MakeBrighterFatterKernelTaskConfig(pexConfig.Config):
    """Config class for bright-fatter effect coefficient calculation."""

    isr = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal""",
    )
    isrMandatorySteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must be performed for valid results. Raises if any of these are False",
        default=['doAssembleCcd']
    )
    isrForbiddenSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must NOT be performed for valid results. Raises if any of these are True",
        default=['doFlat', 'doFringe', 'doAddDistortionModel', 'doBrighterFatter', 'doUseOpticsTransmission',
                 'doUseFilterTransmission', 'doUseSensorTransmission', 'doUseAtmosphereTransmission']
    )
    isrDesirableSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that it is advisable to perform, but are not mission-critical." +
        " WARNs are logged for any of these found to be False.",
        default=['doBias', 'doDark', 'doCrosstalk', 'doDefect', 'doLinearize']
    )
    doCalcGains = pexConfig.Field(
        dtype=bool,
        doc="Measure the per-amplifier gains (using the photon transfer curve method)?",
        default=True,
    )
    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'",
        default='ccd',
    )
    maxIterRegression = pexConfig.Field(
        dtype=int,
        doc="Maximum number of iterations for the regression fitter",
        default=10
    )
    nSigmaClipGainCalc = pexConfig.Field(
        dtype=int,
        doc="Number of sigma to clip the pixel value distribution to during gain calculation",
        default=5
    )
    nSigmaClipRegression = pexConfig.Field(
        dtype=int,
        doc="Number of sigma to clip outliers from the line of best fit to during iterative regression",
        default=3
    )
    xcorrCheckRejectLevel = pexConfig.Field(
        dtype=float,
        doc="Sanity check level for the sum of the input cross-correlations. Arrays which " +
        "sum to greater than this are discarded before the clipped mean is calculated.",
        default=2.0
    )
    maxIterSuccessiveOverRelaxation = pexConfig.Field(
        dtype=int,
        doc="The maximum number of iterations allowed for the successive over-relaxation method",
        default=10000
    )
    eLevelSuccessiveOverRelaxation = pexConfig.Field(
        dtype=float,
        doc="The target residual error for the successive over-relaxation method",
        default=5.0e-14
    )
    nSigmaClipKernelGen = pexConfig.Field(
        dtype=float,
        doc="Number of sigma to clip to during pixel-wise clipping when generating the kernel. See " +
        "the generateKernel docstring for more info.",
        default=4
    )
    nSigmaClipXCorr = pexConfig.Field(
        dtype=float,
        doc="Number of sigma to clip when calculating means for the cross-correlation",
        default=5
    )
    maxLag = pexConfig.Field(
        dtype=int,
        doc="The maximum lag (in pixels) to use when calculating the cross-correlation/kernel",
        default=8
    )
    nPixBorderGainCalc = pexConfig.Field(
        dtype=int,
        doc="The number of border pixels to exclude when calculating the gain",
        default=10
    )
    nPixBorderXCorr = pexConfig.Field(
        dtype=int,
        doc="The number of border pixels to exclude when calculating the cross-correlation and kernel",
        default=10
    )
    biasCorr = pexConfig.Field(
        dtype=float,
        doc="An empirically determined correction factor, used to correct for the sigma-clipping of" +
        " a non-Gaussian distribution. Post DM-15277, code will exist here to calulate appropriate values",
        default=0.9241
    )
    backgroundBinSize = pexConfig.Field(
        dtype=int,
        doc="Size of the background bins",
        default=128
    )
    fixPtcThroughOrigin = pexConfig.Field(
        dtype=bool,
        doc="Constrain the fit of the photon transfer curve to go through the origin when measuring" +
        "the gain?",
        default=True
    )
    level = pexConfig.ChoiceField(
        doc="The level at which to calculate the brighter-fatter kernels",
        dtype=str, default="DETECTOR",
        allowed={
            "AMP": "Every amplifier treated separately",
            "DETECTOR": "One kernel per detector",
        }
    )
    backgroundWarnLevel = pexConfig.Field(
        dtype=float,
        doc="Log warnings if the mean of the fitted background is found to be above this level after " +
        "differencing image pair.",
        default=0.1
    )


class MakeBrighterFatterKernelTaskRunner(pipeBase.TaskRunner):
    """Subclass of TaskRunner for the MakeBrighterFatterKernelTask.

    This transforms the processed arguments generated by the ArgumentParser
    into the arguments expected by makeBrighterFatterKernelTask.run().

    makeBrighterFatterKernelTask.run() takes a two arguments,
    one of which is the dataRef (as usual), and the other is the list
    of visit-pairs, in the form of a list of tuples.
    This list is supplied on the command line as documented,
    and this class parses that, and passes the parsed version
    to the run() method.

    See pipeBase.TaskRunner for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Parse the visit list and pass through explicitly."""
        visitPairs = []
        for visitStringPair in parsedCmd.visitPairs:
            visitStrings = visitStringPair.split(",")
            if len(visitStrings) != 2:
                raise RuntimeError("Found {} visits in {} instead of 2".format(len(visitStrings),
                                                                               visitStringPair))
            try:
                visits = [int(visit) for visit in visitStrings]
            except Exception:
                raise RuntimeError("Could not parse {} as two integer visit numbers".format(visitStringPair))
            visitPairs.append(visits)

        return pipeBase.TaskRunner.getTargetList(parsedCmd, visitPairs=visitPairs, **kwargs)


class BrighterFatterKernelTaskDataIdContainer(pipeBase.DataIdContainer):
    """A DataIdContainer for the MakeBrighterFatterKernelTask."""

    def makeDataRefList(self, namespace):
        """Compute refList based on idList.

        This method must be defined as the dataset does not exist before this
        task is run.

        Parameters
        ----------
        namespace
            Results of parsing the command-line.

        Notes
        -----
        Not called if ``add_id_argument`` called
        with ``doMakeDataRefList=False``.
        Note that this is almost a copy-and-paste of the vanilla implementation,
        but without checking if the datasets already exist,
        as this task exists to make them.
        """
        if self.datasetType is None:
            raise RuntimeError("Must call setDatasetType first")
        butler = namespace.butler
        for dataId in self.idList:
            refList = list(butler.subset(datasetType=self.datasetType, level=self.level, dataId=dataId))
            # exclude nonexistent data
            # this is a recursive test, e.g. for the sake of "raw" data
            if not refList:
                namespace.log.warn("No data found for dataId=%s", dataId)
                continue
            self.refList += refList


class BrighterFatterKernel:
    """A (very) simple class to hold the kernel(s) generated.

    The kernel.kernel is a dictionary holding the kernels themselves.
    One kernel if the level is 'DETECTOR' or,
    nAmps in length, if level is 'AMP'.
    The dict is keyed by either the detector ID or the amplifier IDs.

    The level is the level for which the kernel(s) were generated so that one
    can know how to access the kernels without having to query the shape of
    the dictionary holding the kernel(s).
    """

    def __init__(self, level, kernelDict):
        assert type(level) == str
        assert type(kernelDict) == dict
        if level == 'DETECTOR':
            assert len(kernelDict.keys()) == 1
        if level == 'AMP':
            assert len(kernelDict.keys()) > 1

        self.level = level
        self.kernel = kernelDict


class MakeBrighterFatterKernelTask(pipeBase.CmdLineTask):
    """Brighter-fatter effect correction-kernel calculation task.

    A command line task for calculating the brighter-fatter correction
    kernel from pairs of flat-field images (with the same exposure length).

    The following operations are performed:

    - The configurable isr task is called, which unpersists and assembles the
      raw images, and performs the selected instrument signature removal tasks.
      For the purpose of brighter-fatter coefficient calculation is it
      essential that certain components of isr are *not* performed, and
      recommended that certain others are. The task checks the selected isr
      configuration before it is run, and if forbidden components have been
      selected task will raise, and if recommended ones have not been selected,
      warnings are logged.

    - The gain of the each amplifier in the detector is calculated using
      the photon transfer curve (PTC) method and used to correct the images
      so that all calculations are done in units of electrons, and so that the
      level across amplifier boundaries is continuous.
      Outliers in the PTC are iteratively rejected
      before fitting, with the nSigma rejection level set by
      config.nSigmaClipRegression. Individual pixels are ignored in the input
      images the image based on config.nSigmaClipGainCalc.

    - Each image is then cross-correlated with the one it's paired with
      (with the pairing defined by the --visit-pairs command line argument),
      which is done either the whole-image to whole-image,
      or amplifier-by-amplifier, depending on config.level.

    - Once the cross-correlations have been calculated for each visit pair,
      these are used to generate the correction kernel.
      The maximum lag used, in pixels, and hence the size of the half-size
      of the kernel generated, is given by config.maxLag,
      i.e. a value of 10 will result in a kernel of size 2n-1 = 19x19 pixels.
      Outlier values in these cross-correlations are rejected by using a
      pixel-wise sigma-clipped thresholding to each cross-correlation in
      the visit-pairs-length stack of cross-correlations.
      The number of sigma clipped to is set by config.nSigmaClipKernelGen.

    - Once DM-15277 has been completed, a method will exist to calculate the
      empirical correction factor, config.biasCorr.
      TODO: DM-15277 update this part of the docstring once the ticket is done.
    """

    RunnerClass = MakeBrighterFatterKernelTaskRunner
    ConfigClass = MakeBrighterFatterKernelTaskConfig
    _DefaultName = "makeBrighterFatterKernel"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")

        self.debug = lsstDebug.Info(__name__)
        if self.debug.enabled:
            self.log.info("Running with debug enabled...")
            # If we're displaying, test it works and save displays for later.
            # It's worth testing here as displays are flaky and sometimes
            # can't be contacted, and given processing takes a while,
            # it's a shame to fail late due to display issues.
            if self.debug.display:
                try:
                    afwDisp.setDefaultBackend(self.debug.displayBackend)
                    afwDisp.Display.delAllDisplays()
                    self.disp1 = afwDisp.Display(0, open=True)
                    self.disp2 = afwDisp.Display(1, open=True)

                    im = afwImage.ImageF(1, 1)
                    im.array[:] = [[1]]
                    self.disp1.mtv(im)
                    self.disp1.erase()
                except NameError:
                    self.debug.display = False
                    self.log.warn('Failed to setup/connect to display! Debug display has been disabled')

        plt.interactive(False)  # stop windows popping up when plotting. When headless, use 'agg' backend too
        self.validateIsrConfig()
        self.config.validate()
        self.config.freeze()

    @classmethod
    def _makeArgumentParser(cls):
        """Augment argument parser for the MakeBrighterFatterKernelTask."""
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--visit-pairs", dest="visitPairs", nargs="*",
                            help="Visit pairs to use. Each pair must be of the form INT,INT e.g. 123,456")
        parser.add_id_argument("--id", datasetType="brighterFatterKernel",
                               ContainerClass=BrighterFatterKernelTaskDataIdContainer,
                               help="The ccds to use, e.g. --id ccd=0..100")
        
        return parser

    def validateIsrConfig(self):
        """Check that appropriate ISR settings are being used
        for brighter-fatter kernel calculation."""

        # How should we handle saturation/bad regions?
        # 'doSaturationInterpolation': True
        # 'doNanInterpAfterFlat': False
        # 'doSaturation': True
        # 'doSuspect': True
        # 'doWidenSaturationTrails': True
        # 'doSetBadRegions': True

        configDict = self.config.isr.toDict()

        for configParam in self.config.isrMandatorySteps:
            if configDict[configParam] is False:
                raise RuntimeError('Must set config.isr.%s to True '
                                   'for brighter-fatter kernel calulation' % configParam)

        for configParam in self.config.isrForbiddenSteps:
            if configDict[configParam] is True:
                raise RuntimeError('Must set config.isr.%s to False '
                                   'for brighter-fatter kernel calulation' % configParam)

        for configParam in self.config.isrDesirableSteps:
            if configParam not in configDict:
                self.log.info('Failed to find key %s in the isr config dict. You probably want ' +
                              'to set the equivalent for your obs_package to True.' % configParam)
                continue
            if configDict[configParam] is False:
                self.log.warn('Found config.isr.%s set to False for brighter-fatter kernel calulation. '
                              'It is probably desirable to have this set to True' % configParam)

        # subtask settings
        if not self.config.isr.assembleCcd.doTrim:
            raise RuntimeError('Must trim when assembling CCDs. Set config.isr.assembleCcd.doTrim to True')

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, visitPairs):
        """Run the brighter-fatter measurement task.

        For a dataRef (which is each detector here),
        and given a list of visit pairs, calulate the
        brighter-fatter kernel for the detector.

        Parameters
        ----------
        dataRef : list of lsst.daf.persistence.ButlerDataRef
            dataRef for the detector for the visits to be fit.
        visitPairs : `iterable` of `tuple` of `int`
            Pairs of visit numbers to be processed together
        """
        xcorrs = {}  # dict of lists keyed by either amp or detector depending on config.level
        means = {}
        kernels = {}
        print("Starting BF kernel task - 10-Jan-19")
        # setup necessary objects
        detNum = dataRef.dataId[self.config.ccdKey]
        detector = dataRef.get('camera')[dataRef.dataId[self.config.ccdKey]]
        ampInfoCat = detector.getAmpInfoCatalog()
        ampNames = [amp.getName() for amp in ampInfoCat]

        if self.config.level == 'DETECTOR':
            xcorrs = {detNum: []}
            means = {detNum: []}
        elif self.config.level == 'AMP':
            # NB: don't use dataRef.get('raw_detector')
            # this currently doesn't work for composites because of the way
            # composite objects (i.e. LSST images) are handled/constructed
            # these need to be retrieved from the camera and dereferenced
            # rather than accessed directly
            xcorrs = {key: [] for key in ampNames}
            means = {key: [] for key in ampNames}
        else:
            raise RuntimeError("Unsupported level: {}".format(self.config.level))

        # calculate or retrieve the gains
        if self.config.doCalcGains:
            # In this case, we set the gains = 1.0 for now, and calculate them later after
            # the cross-corrrelations have been calculated. This is only supported when the
            # level is by AMP.
            if self.config.level == 'DETECTOR':
                raise RuntimeError('doCalcGains = True is inconsistent with config.level = DETECTOR')                
            self.log.info('Setting gain to 1.0 for detector %s' % detNum)
            gains = {key: 1.0 for key in ampNames}
            dataRef.put(gains, datasetType='brighterFatterGain')

        else:
            gains = dataRef.get('brighterFatterGain')
            if not gains:
                raise RuntimeError('Failed to retrieve gains for detector %s' % detNum)
            self.log.info('Retrieved stored gain for detector %s' % detNum)
        self.log.debug('Detector %s has gains %s' % (detNum, gains))

        
        # Loop over pairs of visits
        # calculating the cross-correlations at the required level
        for (v1, v2) in visitPairs:
            print("Running pair ", v1, v2)            
            dataRef.dataId['visit'] = v1
            exp1 = self.isr.runDataRef(dataRef).exposure
            dataRef.dataId['visit'] = v2
            exp2 = self.isr.runDataRef(dataRef).exposure
            del dataRef.dataId['visit']
            self._checkExpLengthEqual(exp1, exp2, v1, v2)

            self.log.info('Preparing images for cross-correlation calculation for detector %s' % detNum)
            # note the shape of these returns depends on level
            _scaledMaskedIms1, _means1 = self._makeCroppedExposures(exp1, gains, self.config.level)
            _scaledMaskedIms2, _means2 = self._makeCroppedExposures(exp2, gains, self.config.level)

            # Compute the cross-correlation and means
            # at the appropriate config.level:
            # - "DETECTOR": one key, so compare the two visits to each other
            # - "AMP": n_amp keys, comparing each amplifier of one visit
            #          to the same amplifier in the visit its paired with
            for det_object in _scaledMaskedIms1.keys():
                _xcorr, _ = self._crossCorrelate(_scaledMaskedIms1[det_object],
                                                 _scaledMaskedIms2[det_object])
                xcorrs[det_object].append(_xcorr)
                means[det_object].append([_means1[det_object], _means2[det_object]])

                # TODO: DM-15305 improve debug functionality here.
                # This is position 1 for the removed code.
        # Can't get this to work.  Pickling the data instead.
        # I tried adding these to obs_lsstCam/policy/lsstCamMapper.yaml, but that still didn't work.
        # Also added them to obs_base/policy/datasets.yaml, but it still doesn't work.
        #dataRef.put(means, "brighterFatterMeans")
        #dataRef.put(xcorrs, "brighterFatterXcorrs")        
        corr_pickle = {'xcorrs':xcorrs, 'means': means}
        filename ='corr_data_%s_full.pkl'%detNum
        with open(filename, 'wb') as f:
            pkl.dump(corr_pickle, f)

        if self.config.doCalcGains:
            # Now we calculate and apply the gains to the calculated
            # means and cross-correlations
            self.log.info('Calculating gains for detector %s' % detNum)
            means, xcorrs  = self._calculateAndApplyGains(dataRef, means, xcorrs)           
            self.log.debug('Finished gain estimation for detector %s' % detNum)

        # generate the kernel(s)
        self.log.info('Generating kernel(s) for %s' % detNum)
        for det_object in xcorrs.keys():  # looping over either detectors or amps
            if self.config.level == 'DETECTOR':
                objId = 'detector %s' % det_object
            elif self.config.level == 'AMP':
                objId = 'detector %s AMP %s' % (detNum, det_object)
            kernels[det_object] = self.generateKernel(xcorrs[det_object], means[det_object], objId)
        dataRef.put(BrighterFatterKernel(self.config.level, kernels))
        
        self.log.info('Finished generating kernel(s) for %s' % detNum)
        return pipeBase.Struct(exitStatus=0)

    def _makeCroppedExposures(self, exp, gains, level):
        """Prepare exposure for cross-correlation calculation.

        For each amp, crop by the border amount, specified by
        config.nPixBorderXCorr, then rescale by the gain
        and subtract the sigma-clipped mean.
        If the level is 'DETECTOR' then this is done
        to the whole image so that it can be cross-correlated, with a copy
        being returned.
        If the level is 'AMP' then this is done per-amplifier,
        and a copy of each prepared amp-image returned.

        Parameters:
        -----------
        exp : `lsst.afw.image.exposure.ExposureF`
            The exposure to prepare
        gains : `dict` of `float`
            Dictionary of the amplifier gain values, keyed by amplifier name
        level : `str`
            Either `AMP` or `DETECTOR`

        Returns:
        --------
        scaledMaskedIms : `dict` of `lsst.afw.image.maskedImage.MaskedImageF`
            Depending on level, this is either one item, or n_amp items,
            keyed by detectorId or ampName

        Notes:
        ------
        This function is controlled by the following config parameters:
        nPixBorderXCorr : `int`
            The number of border pixels to exclude
        nSigmaClipXCorr : `float`
            The number of sigma to be clipped to
        """
        assert(isinstance(exp, afwImage.ExposureF))

        local_exp = exp.clone()  # we don't want to modify the image passed in
        del exp  # ensure we don't make mistakes!

        border = self.config.nPixBorderXCorr
        sigma = self.config.nSigmaClipXCorr

        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(sigma)

        means = {}
        returnAreas = {}

        detector = local_exp.getDetector()
        ampInfoCat = detector.getAmpInfoCatalog()

        mi = local_exp.getMaskedImage()  # makeStatistics does not seem to take exposures
        temp = mi.clone()

        # Rescale each amp by the appropriate gain and subtract the mean.
        # NB these are views modifying the image in-place
        for amp in ampInfoCat:
            ampName = amp.getName()
            rescaleIm = mi[amp.getBBox()]  # the soon-to-be scaled, mean subtractedm, amp image
            rescaleTemp = temp[amp.getBBox()]
            mean = afwMath.makeStatistics(rescaleIm, afwMath.MEANCLIP, sctrl).getValue()
            gain = gains[ampName]
            rescaleIm *= gain
            rescaleTemp *= gain
            self.log.debug("mean*gain = %s, clipped mean = %s" %
                           (mean*gain, afwMath.makeStatistics(rescaleIm, afwMath.MEANCLIP,
                                                              sctrl).getValue()))
            rescaleIm -= mean*gain

            if level == 'AMP':  # build the dicts if doing amp-wise
                means[ampName] = afwMath.makeStatistics(rescaleTemp[border: -border, border: -border,
                                                        afwImage.LOCAL], afwMath.MEANCLIP, sctrl).getValue()
                returnAreas[ampName] = rescaleIm

        if level == 'DETECTOR':  # else just average the whole detector
            detName = local_exp.getDetector().getId()
            means[detName] = afwMath.makeStatistics(temp[border: -border, border: -border, afwImage.LOCAL],
                                                    afwMath.MEANCLIP, sctrl).getValue()
            returnAreas[detName] = rescaleIm

        return returnAreas, means

    def _crossCorrelate(self, maskedIm0, maskedIm1, runningBiasCorrSim=False, frameId=None, detId=None):
        """Calculate the cross-correlation of an area.

        If the area in question contains multiple amplifiers then they must
        have been gain corrected.

        Parameters:
        -----------
        maskedIm0 : `lsst.afw.image.MaskedImageF`
            The first image area
        maskedIm1 : `lsst.afw.image.MaskedImageF`
            The first image area
        frameId : `str`, optional
            The frame identifier for use in the filename
            if writing debug outputs.
        detId : `str`, optional
            The detector identifier (detector, or detector+amp,
            depending on config.level) for use in the filename
            if writing debug outputs.
        runningBiasCorrSim : `bool`
            Set to true when using this function to calculate the amount of bias
            introduced by the sigma clipping. If False, the biasCorr parameter
            is divided by to remove the bias, but this is, of course, not
            appropriate when this is the parameter being measured.

        Returns:
        --------
        xcorr : `np.ndarray`
            The quarter-image cross-correlation
        mean : `float`
            The sum of the means of the input images,
            sigma-clipped, and with borders applied.
            This is used when using this function with simulations to calculate
            the biasCorr parameter.

        Notes:
        ------
        This function is controlled by the following config parameters:
        maxLag : `int`
            The maximum lag to use in the cross-correlation calculation
        nPixBorderXCorr : `int`
            The number of border pixels to exclude
        nSigmaClipXCorr : `float`
            The number of sigma to be clipped to
        biasCorr : `float`
            Parameter used to correct from the bias introduced
            by the sigma cuts.
        """
        maxLag = self.config.maxLag
        border = self.config.nPixBorderXCorr
        sigma = self.config.nSigmaClipXCorr
        biasCorr = self.config.biasCorr

        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(sigma)

        mean = afwMath.makeStatistics(maskedIm0.getImage()[border: -border, border: -border, afwImage.LOCAL],
                                      afwMath.MEANCLIP, sctrl).getValue()
        mean += afwMath.makeStatistics(maskedIm1.getImage()[border: -border, border: -border, afwImage.LOCAL],
                                       afwMath.MEANCLIP, sctrl).getValue()

        # Diff the images, and apply border
        diff = maskedIm0.clone()
        diff -= maskedIm1.getImage()
        diff = diff[border: -border, border: -border, afwImage.LOCAL]

        if self.debug.writeDiffImages:
            filename = '_'.join(['diff', 'detector', detId, frameId, '.fits'])
            diff.writeFits(os.path.join(self.debug.debugDataPath, filename))

        # Subtract background.  It should be a constant, but it isn't always
        binsize = self.config.backgroundBinSize
        nx = diff.getWidth()//binsize
        ny = diff.getHeight()//binsize
        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
        bkgd = afwMath.makeBackground(diff, bctrl)
        bgImg = bkgd.getImageF(afwMath.Interpolate.CUBIC_SPLINE, afwMath.REDUCE_INTERP_ORDER)
        bgMean = np.mean(bgImg.getArray())
        if abs(bgMean) >= self.config.backgroundWarnLevel:
            self.log.warn('Mean of background = %s > config.maxBackground' % bgMean)

        diff -= bgImg

        if self.debug.writeDiffImages:
            filename = '_'.join(['bgSub', 'diff', 'detector', detId, frameId, '.fits'])
            diff.writeFits(os.path.join(self.debug.debugDataPath, filename))
        if self.debug.display:
            self.disp1.mtv(diff, title=frameId)

        self.log.debug("Median and variance of diff:")
        self.log.debug("%s" % afwMath.makeStatistics(diff, afwMath.MEDIAN, sctrl).getValue())
        self.log.debug("%s" % afwMath.makeStatistics(diff, afwMath.VARIANCECLIP,
                                                     sctrl).getValue(), np.var(diff.getImage().getArray()))

        # Measure the correlations
        dim0 = diff[0: -maxLag, : -maxLag, afwImage.LOCAL]
        dim0 -= afwMath.makeStatistics(dim0, afwMath.MEANCLIP, sctrl).getValue()
        width, height = dim0.getDimensions()
        xcorr = np.zeros((maxLag + 1, maxLag + 1), dtype=np.float64)

        for xlag in range(maxLag + 1):
            for ylag in range(maxLag + 1):
                dim_xy = diff[xlag:xlag + width, ylag: ylag + height, afwImage.LOCAL].clone()
                dim_xy -= afwMath.makeStatistics(dim_xy, afwMath.MEANCLIP, sctrl).getValue()
                dim_xy *= dim0
                xcorr[xlag, ylag] = afwMath.makeStatistics(dim_xy, afwMath.MEANCLIP, sctrl).getValue()
                if not runningBiasCorrSim:
                    xcorr[xlag, ylag] /= biasCorr

        # TODO: DM-15305 improve debug functionality here.
        # This is position 2 for the removed code.

        return xcorr, mean

    def _calculateAndApplyGains(self, dataRef, means, xcorrs):
        """Estimate the amplifier gains using the calculated means and variances.

        Given a dataRef and the calculated means and variances,
        calculate the gain for each amplifier in the detector
        using the photon transfer curve (PTC) method.

        The gain is calculated as the linear part of the 
        photon transfer curve only.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butler.Butler.dataRef`
            dataRef for the detector for the flats to be used
        means : `list` of mean values of the two visit pairs
        xcorrs : `list` of cross-correlations values of the two visit pairs

        Returns
        -------
        means : `list` of mean values of the two visit pairs
        xcorrs : `list` of cross-correlations values of the two visit pairs

        Both of these have been corrected for the measured gains
        The gains are also stored in dataRef datasetType='brighterFatterGain'.

        """
        print("In calculateAndApplyGains")                    
        # NB: don't use dataRef.get('raw_detector') due to composites
        detector = dataRef.get('camera')[dataRef.dataId[self.config.ccdKey]]

        gains = {}
        gain_adjusted_means = {}
        gain_adjusted_xcorrs = {}
        for amp in detector:
            ampName = amp.getName()
            print("Doing Amp ", ampName)                                
            ampMeans = []
            ampVariances = []
            for i in range(len(means[ampName])):
                # The division by 2 below is to calculate the average flux from the two visits
                ampMeans.append((means[ampName][i][0] + means[ampName][i][1]) / 2.0)
                # The division by 2 below is because the variance of a difference is twice the variance of each visit
                ampVariances.append(xcorrs[ampName][i][0,0] / 2.0)
            # Now fit a cubic polynomial to the PTC
            # and use the linear part as the gain
            ptc_coefs = np.polyfit(ampMeans, ampVariances, 3)
            slopeToUse = ptc_coefs[2]
            gain = 1.0 / slopeToUse
            #if self.debug.enabled:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title("Photon Transfer Curve %s"%ampName,fontsize=24)
            x_values = np.asarray(ampMeans)
            ax.plot(x_values,
                    np.asarray(ampVariances), linestyle='None', color = 'green', marker='x', label='data')
            cubic_fit = ptc_coefs[0]*x_values*x_values*x_values + ptc_coefs[1]*x_values*x_values +\
                        ptc_coefs[2]*x_values + ptc_coefs[3]
            ax.plot(x_values,
                    cubic_fit, linestyle='--', color='green', label='Cubic fit')
            ax.plot(x_values,
                    x_values*slopeToUse, color='red', label='Linear fit')
            ax.set_xlabel("Flux(ADU)", fontsize=18)
            ax.set_ylabel("Variance(ADU^2)",fontsize=18)
            ax.legend()
            dataRef.put(fig, "plotBrighterFatterPtc", amp=ampName)
            self.log.info('Saved PTC for detector %s amp %s' % (detector.getId(), ampName))
            gains[ampName] = gain
            print("Doing Amp ", ampName, ptc_coefs)                                                        
            print("Doing Amp ", ampName, "Gain = ", gain)                                            
            gain_adjusted_means[ampName] = [[i*gain for i in pair] for pair in means[ampName]]
            gain_adjusted_xcorrs[ampName] = [arr*gain*gain for arr in xcorrs[ampName]]
        plt.close('all')
        dataRef.put(gains, datasetType='brighterFatterGain')
        return gain_adjusted_means, gain_adjusted_xcorrs

    @staticmethod
    def _checkExpLengthEqual(exp1, exp2, v1=None, v2=None):
        """Check the exposure lengths of two exposures are equal.

        Parameters:
        -----------
        exp1 : `lsst.afw.image.exposure.ExposureF`
            First exposure to check
        exp2 : `lsst.afw.image.exposure.ExposureF`
            Second exposure to check
        v1 : `int` or `str`, optional
            First visit of the visit pair
        v2 : `int` or `str`, optional
            Second visit of the visit pair

        Raises:
        -------
        RuntimeError
            Raised if the exposure lengths of the two exposures are not equal
        """
        expTime1 = exp1.getInfo().getVisitInfo().getExposureTime()
        expTime2 = exp2.getInfo().getVisitInfo().getExposureTime()
        if expTime1 != expTime2:
            msg = "Exposure lengths for visit pairs must be equal. " + \
                  "Found %s and %s" % (expTime1, expTime2)
            if v1 and v2:
                msg += " for visit pair %s, %s" % (v1, v2)
            raise RuntimeError(msg)


    def _plotXcorr(self, xcorr, mean, zmax=0.05, title=None, fig=None, saveToFileName=None):
        """Plot the correlation functions."""
        try:
            xcorr = xcorr.getArray()
        except Exception:
            pass

        xcorr /= float(mean)
        # xcorr.getArray()[0,0]=abs(xcorr.getArray()[0,0]-1)

        if fig is None:
            fig = plt.figure()
        else:
            fig.clf()

        ax = fig.add_subplot(111, projection='3d')
        ax.azim = 30
        ax.elev = 20

        nx, ny = np.shape(xcorr)

        xpos, ypos = np.meshgrid(np.arange(nx), np.arange(ny))
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(nx*ny)
        dz = xcorr.flatten()
        dz[dz > zmax] = zmax

        ax.bar3d(xpos, ypos, zpos, 1, 1, dz, color='b', zsort='max', sort_zpos=100)
        if xcorr[0, 0] > zmax:
            ax.bar3d([0], [0], [zmax], 1, 1, 1e-4, color='c')

        ax.set_xlabel("row")
        ax.set_ylabel("column")
        ax.set_zlabel(r"$\langle{(F_i - \bar{F})(F_i - \bar{F})}\rangle/\bar{F}$")

        if title:
            fig.suptitle(title)
        if saveToFileName:
            fig.savefig(saveToFileName)

    def _iterativeRegression(self, x, y, fixThroughOrigin=False, nSigmaClip=None, maxIter=None):
        """Use linear regression to fit a line, iteratively removing outliers.

        Useful when you have a sufficiently large numbers of points on your PTC.
        This function iterates until either there are no outliers of
        config.nSigmaClip magnitude, or until the specified maximum number
        of iterations has been performed.

        Parameters:
        -----------
        x : `numpy.array`
            The independent variable. Must be a numpy array, not a list.
        y : `numpy.array`
            The dependent variable. Must be a numpy array, not a list.
        fixThroughOrigin : `bool`, optional
            Whether to fix the PTC through the origin or allow an y-intercept.
        nSigmaClip : `float`, optional
            The number of sigma to clip to.
            Taken from the task config if not specified.
        maxIter : `int`, optional
            The maximum number of iterations allowed.
            Taken from the task config if not specified.

        Returns:
        --------
        slope : `float`
            The slope of the line of best fit
        intercept : `float`
            The y-intercept of the line of best fit
        """
        if not maxIter:
            maxIter = self.config.maxIterRegression
        if not nSigmaClip:
            nSigmaClip = self.config.nSigmaClipRegression

        nIter = 0
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(nSigmaClip)

        if fixThroughOrigin:
            while nIter < maxIter:
                nIter += 1
                self.log.debug("Origin fixed, iteration # %s using %s elements:" % (nIter, np.shape(x)[0]))
                TEST = x[:, np.newaxis]
                slope, _, _, _ = np.linalg.lstsq(TEST, y)
                slope = slope[0]
                res = y - slope * x
                resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
                resStd = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
                index = np.where((res > (resMean + nSigmaClip*resStd)) |
                                 (res < (resMean - nSigmaClip*resStd)))
                self.log.debug("%.3f %.3f %.3f %.3f" % (resMean, resStd, np.max(res), nSigmaClip))
                if np.shape(np.where(index))[1] == 0 or (nIter >= maxIter):  # run out of points or iters
                    break
                x = np.delete(x, index)
                y = np.delete(y, index)

            return slope, 0

        while nIter < maxIter:
            nIter += 1
            self.log.debug("Iteration # %s using %s elements:" % (nIter, np.shape(x)[0]))
            xx = np.vstack([x, np.ones(len(x))]).T
            ret, _, _, _ = np.linalg.lstsq(xx, y)
            slope, intercept = ret
            res = y - slope*x - intercept
            resMean = afwMath.makeStatistics(res, afwMath.MEANCLIP, sctrl).getValue()
            resStd = np.sqrt(afwMath.makeStatistics(res, afwMath.VARIANCECLIP, sctrl).getValue())
            index = np.where((res > (resMean + nSigmaClip * resStd)) | (res < resMean - nSigmaClip * resStd))
            self.log.debug("%.3f %.3f %.3f %.3f" % (resMean, resStd, np.max(res), nSigmaClip))
            if np.shape(np.where(index))[1] == 0 or (nIter >= maxIter):  # run out of points, or iterations
                break
            x = np.delete(x, index)
            y = np.delete(y, index)

        return slope, intercept

    def generateKernel(self, corrs, means, objId, rejectLevel=None):
        """Generate the full kernel from a list of cross-correlations and means.

        Taking a list of quarter-image, gain-corrected cross-correlations,
        do a pixel-wise sigma-clipped mean of each,
        and tile into the full-sized kernel image.

        Each corr in corrs is one quarter of the full cross-correlation,
        and has been gain-corrected. Each mean in means is a tuple of the means
        of the two individual images, corresponding to that corr.

        Parameters:
        -----------
        corrs : `list` of `numpy.ndarray`, (Ny, Nx)
            A list of the quarter-image cross-correlations
        means : `dict` of `tuples` of `floats`
            The means of the input images for each corr in corrs
        rejectLevel : `float`, optional
            This is essentially is a sanity check parameter.
            If this condition is violated there is something unexpected
            going on in the image, and it is discarded from the stack before
            the clipped-mean is calculated.
            If not provided then config.xcorrCheckRejectLevel is used

        Returns:
        --------
        kernel : `numpy.ndarray`, (Ny, Nx)
            The output kernel
        """
        if not rejectLevel:
            rejectLevel = self.config.xcorrCheckRejectLevel

        # Try to average over a set of possible inputs.
        # This generates a simple function of the kernel that
        # should be constant across the images, and averages that.
        xcorrList = []
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(self.config.nSigmaClipKernelGen)

        for corrNum, ((mean1, mean2), corr) in enumerate(zip(means, corrs)):
            corr[0, 0] -= (mean1 + mean2)
            if corr[0, 0] > 0:
                self.log.warn('Skipped item %s due to unexpected value of (variance-mean)' % corrNum)
                continue
            corr /= -1.0*(mean1**2 + mean2**2)

            fullCorr = self._tileArray(corr)

            xcorrCheck = np.abs(np.sum(fullCorr))/np.sum(np.abs(fullCorr))
            if xcorrCheck > rejectLevel:
                self.log.warn("Sum of the xcorr is unexpectedly high. Investigate item num %s for %s. \n"
                              "value = %s" % (corrNum, objId, xcorrCheck))
                continue
            xcorrList.append(fullCorr)

        if not xcorrList:
            raise RuntimeError("Cannot generate kernel because all inputs were discarded. "
                               "Either the data is bad, or config.xcorrCheckRejectLevel is too low")

        # stack the individual xcorrs and apply a per-pixel clipped-mean
        meanXcorr = np.zeros_like(fullCorr)
        xcorrList = np.transpose(xcorrList)
        for i in range(np.shape(meanXcorr)[0]):
            for j in range(np.shape(meanXcorr)[1]):
                meanXcorr[i, j] = afwMath.makeStatistics(xcorrList[i, j], afwMath.MEANCLIP, sctrl).getValue()

        return self.successiveOverRelax(meanXcorr)

    def successiveOverRelax(self, source, maxIter=None, eLevel=None):
        """An implementation of the successive over relaxation (SOR) method.

        A numerical method for solving a system of linear equations
        with faster convergence than the Gauss-Seidel method.

        Parameters:
        -----------
        source : `numpy.ndarray`
            The input array
        maxIter : `int`, optional
            Maximum number of iterations to attempt before aborting
        eLevel : `float`, optional
            The target error level at which we deem convergence to have occured

        Returns:
        --------
        output : `numpy.ndarray`
            The solution
        """
        if not maxIter:
            maxIter = self.config.maxIterSuccessiveOverRelaxation
        if not eLevel:
            eLevel = self.config.eLevelSuccessiveOverRelaxation

        assert source.shape[0] == source.shape[1], "Input array must be square"
        # initialise, and set boundary conditions
        func = np.zeros([source.shape[0] + 2, source.shape[1] + 2])
        resid = np.zeros([source.shape[0] + 2, source.shape[1] + 2])
        rhoSpe = np.cos(np.pi/source.shape[0])  # Here a square grid is assummed

        # Calculate the initial error
        for i in range(1, func.shape[0] - 1):
            for j in range(1, func.shape[1] - 1):
                resid[i, j] = (func[i, j - 1] + func[i, j + 1] + func[i - 1, j] +
                               func[i + 1, j] - 4*func[i, j] - source[i - 1, j - 1])
        inError = np.sum(np.abs(resid))

        # Iterate until convergence
        # We perform two sweeps per cycle,
        # updating 'odd' and 'even' points separately
        nIter = 0
        omega = 1.0
        dx = 1.0
        while nIter < maxIter*2:
            outError = 0
            if nIter%2 == 0:
                for i in range(1, func.shape[0] - 1, 2):
                    for j in range(1, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j-1] + func[i, j + 1] + func[i - 1, j] +
                                            func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0] - 1, 2):
                    for j in range(2, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j] +
                                            func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
            else:
                for i in range(1, func.shape[0] - 1, 2):
                    for j in range(2, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j] +
                                            func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0] - 1, 2):
                    for j in range(1, func.shape[1] - 1, 2):
                        resid[i, j] = float(func[i, j - 1] + func[i, j + 1] + func[i - 1, j] +
                                            func[i + 1, j] - 4.0*func[i, j] - dx*dx*source[i - 1, j - 1])
                        func[i, j] += omega*resid[i, j]*.25
            outError = np.sum(np.abs(resid))
            if outError < inError*eLevel:
                break
            if nIter == 0:
                omega = 1.0/(1 - rhoSpe*rhoSpe/2.0)
            else:
                omega = 1.0/(1 - rhoSpe*rhoSpe*omega/4.0)
            nIter += 1

        if nIter >= maxIter*2:
            self.log.warn("Failure: SuccessiveOverRelaxation did not converge in %s iterations."
                          "\noutError: %s, inError: %s," % (nIter//2, outError, inError*eLevel))
        else:
            self.log.info("Success: SuccessiveOverRelaxation converged in %s iterations."
                          "\noutError: %s, inError: %s", nIter//2, outError, inError*eLevel)
        return func[1: -1, 1: -1]

    @staticmethod
    def _tileArray(in_array):
        """Given an input quarter-image, tile/mirror it and return full image.

        Given a square input of side-length n, of the form

        input = array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

        return an array of size 2n-1 as

        output = array([[ 9,  8,  7,  8,  9],
                        [ 6,  5,  4,  5,  6],
                        [ 3,  2,  1,  2,  3],
                        [ 6,  5,  4,  5,  6],
                        [ 9,  8,  7,  8,  9]])

        Parameters:
        -----------
        input : `np.array`
            The square input quarter-array

        Returns:
        --------
        output : `np.array`
            The full, tiled array
        """
        assert(in_array.shape[0] == in_array.shape[1])
        length = in_array.shape[0] - 1
        output = np.zeros((2*length + 1, 2*length + 1))

        for i in range(length + 1):
            for j in range(length + 1):
                output[i + length, j + length] = in_array[i, j]
                output[-i + length, j + length] = in_array[i, j]
                output[i + length, -j + length] = in_array[i, j]
                output[-i + length, -j + length] = in_array[i, j]
        return output

    @staticmethod
    def _convertImagelikeToFloatImage(imagelikeObject):
        """Turn an exposure or masked image of any type into an ImageF."""
        for attr in ("getMaskedImage", "getImage"):
            if hasattr(imagelikeObject, attr):
                imagelikeObject = getattr(imagelikeObject, attr)()
        try:
            floatImage = imagelikeObject.convertF()
        except AttributeError:
            raise RuntimeError("Failed to convert image to float")
        return floatImage


def calcBiasCorr(fluxLevels, imageShape, repeats=1, seed=0, addCorrelations=False,
                 correlationStrength=0.1, maxLag=10, nSigmaClip=5, border=10):
    """Calculate the bias induced when sigma-clipping non-Gassian distributions.

    Fill image-pairs of the specified size with Poisson-distributed values,
    adding correlations as necessary. Then calculate the cross correlation,
    and calculate the bias induced using the cross-correlation image
    and the image means.

    Parameters:
    -----------
    fluxLevels : `list` of `int`
        The mean flux levels at which to simiulate.
        Nominal values might be something like [70000, 90000, 110000]
    imageShape : `tuple` of `int`
        The shape of the image array to simulate, nx by ny pixels.
    repeats : `int`, optional
        Number of repeats to perform so that results
        can be averaged to improve SNR.
    seed : `int`, optional
        The random seed to use for the Poisson points.
    addCorrelations : `bool`, optional
        Whether to add brighter-fatter-like correlations to the simulated images
        If true, a correlation between x_{i,j} and x_{i+1,j+1} is introduced
        by adding a*x_{i,j} to x_{i+1,j+1}
    correlationStrength : `float`, optional
        The strength of the correlations.
        This is the value of the coefficient `a` in the above definition.
    maxLag : `int`, optional
        The maximum lag to work to in pixels
    nSigmaClip : `float`, optional
        Number of sigma to clip to when calculating the sigma-clipped mean.
    border : `int`, optional
        Number of border pixels to mask

    Returns:
    --------
    biases : `dict` of `list` of `float`
        A dictionary, keyed by flux level, containing a list of the biases
        for each repeat at that flux level
    means : `dict` of `list` of `float`
        A dictionary, keyed by flux level, containing a list of the average mean
        fluxes (average of the mean of the two images)
        for the image pairs at that flux level
    xcorrs : `dict` of `list` of `np.ndarray`
        A dictionary, keyed by flux level, containing a list of the xcorr
        images for the image pairs at that flux level
    """
    means = {f: [] for f in fluxLevels}
    xcorrs = {f: [] for f in fluxLevels}
    biases = {f: [] for f in fluxLevels}

    config = MakeBrighterFatterKernelTaskConfig()
    config.isrMandatorySteps = []  # no isr but the validation routine is still run
    config.isrForbiddenSteps = []
    config.nSigmaClipXCorr = nSigmaClip
    config.nPixBorderXCorr = border
    config.maxLag = maxLag
    task = MakeBrighterFatterKernelTask(config=config)

    im0 = afwImage.maskedImage.MaskedImageF(imageShape[1], imageShape[0])
    im1 = afwImage.maskedImage.MaskedImageF(imageShape[1], imageShape[0])

    random = np.random.RandomState(seed)

    for rep in range(repeats):
        for flux in fluxLevels:
            data0 = random.poisson(flux, (imageShape)).astype(float)
            data1 = random.poisson(flux, (imageShape)).astype(float)
            if addCorrelations:
                data0[1:, 1:] += correlationStrength*data0[: -1, : -1]
                data1[1:, 1:] += correlationStrength*data1[: -1, : -1]
            im0.image.array[:, :] = data0
            im1.image.array[:, :] = data1

            _xcorr, _means = task._crossCorrelate(im0, im1, runningBiasCorrSim=True)

            means[flux].append(_means)
            xcorrs[flux].append(_xcorr)
            if addCorrelations:
                bias = xcorrs[flux][-1][1, 1]/means[flux][-1]*(1 + correlationStrength)/correlationStrength
                print("Simulated/expected avg. flux: %.1f, %.1f" % (flux, means[flux][-1]/2))
                print("Bias: %.6f" % bias)
            else:
                bias = xcorrs[flux][-1][0, 0]/means[flux][-1]
                print("Simulated/expected avg. flux: %.1f, %.1f" % (flux, means[flux][-1]/2))
                print("Bias: %.6f" % bias)
            biases[flux].append(bias)

    return biases, means, xcorrs
