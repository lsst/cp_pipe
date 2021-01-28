# This file is part of cp_pipe.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
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
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

__all__ = ['PairedVisitListTaskRunner', 'SingleVisitListTaskRunner',
           'NonexistentDatasetTaskDataIdContainer', 'parseCmdlineNumberString',
           'countMaskedPixels', 'checkExpLengthEqual', 'ddict2dict']

import re
import numpy as np
from scipy.optimize import leastsq
import numpy.polynomial.polynomial as poly

import lsst.pipe.base as pipeBase
import lsst.ip.isr as ipIsr
from lsst.ip.isr import isrMock
import lsst.log
import lsst.afw.image

import galsim


def calculateWeightedReducedChi2(measured, model, weightsMeasured, nData, nParsModel):
    """Calculate weighted reduced chi2.

    Parameters
    ----------

    measured : `list`
        List with measured data.

    model : `list`
        List with modeled data.

    weightsMeasured : `list`
        List with weights for the measured data.

    nData : `int`
        Number of data points.

    nParsModel : `int`
        Number of parameters in the model.

    Returns
    -------

    redWeightedChi2 : `float`
        Reduced weighted chi2.
    """

    wRes = (measured - model)*weightsMeasured
    return ((wRes*wRes).sum())/(nData-nParsModel)


def makeMockFlats(expTime, gain=1.0, readNoiseElectrons=5, fluxElectrons=1000,
                  randomSeedFlat1=1984, randomSeedFlat2=666, powerLawBfParams=[],
                  expId1=0, expId2=1):
    """Create a pair or mock flats with isrMock.

    Parameters
    ----------
    expTime : `float`
        Exposure time of the flats.

    gain : `float`, optional
        Gain, in e/ADU.

    readNoiseElectrons : `float`, optional
        Read noise rms, in electrons.

    fluxElectrons : `float`, optional
        Flux of flats, in electrons per second.

    randomSeedFlat1 : `int`, optional
        Random seed for the normal distrubutions for the mean signal and noise (flat1).

    randomSeedFlat2 : `int`, optional
        Random seed for the normal distrubutions for the mean signal and noise (flat2).

    powerLawBfParams : `list`, optional
        Parameters for `galsim.cdmodel.PowerLawCD` to simulate the brightter-fatter effect.

    expId1 : `int`, optional
        Exposure ID for first flat.

    expId2 : `int`, optional
        Exposure ID for second flat.

    Returns
    -------

    flatExp1 : `lsst.afw.image.exposure.exposure.ExposureF`
        First exposure of flat field pair.

    flatExp2 : `lsst.afw.image.exposure.exposure.ExposureF`
        Second exposure of flat field pair.

    Notes
    -----
    The parameters of `galsim.cdmodel.PowerLawCD` are `n, r0, t0, rx, tx, r, t, alpha`. For more
    information about their meaning, see the Galsim documentation
    https://galsim-developers.github.io/GalSim/_build/html/_modules/galsim/cdmodel.html
    and Gruen+15 (1501.02802).

    Example: galsim.cdmodel.PowerLawCD(8, 1.1e-7, 1.1e-7, 1.0e-8, 1.0e-8, 1.0e-9, 1.0e-9, 2.0)
    """
    flatFlux = fluxElectrons  # e/s
    flatMean = flatFlux*expTime  # e
    readNoise = readNoiseElectrons  # e

    mockImageConfig = isrMock.IsrMock.ConfigClass()

    mockImageConfig.flatDrop = 0.99999
    mockImageConfig.isTrimmed = True

    flatExp1 = isrMock.FlatMock(config=mockImageConfig).run()
    flatExp2 = flatExp1.clone()
    (shapeY, shapeX) = flatExp1.getDimensions()
    flatWidth = np.sqrt(flatMean)

    rng1 = np.random.RandomState(randomSeedFlat1)
    flatData1 = rng1.normal(flatMean, flatWidth, (shapeX, shapeY)) + rng1.normal(0.0, readNoise,
                                                                                 (shapeX, shapeY))
    rng2 = np.random.RandomState(randomSeedFlat2)
    flatData2 = rng2.normal(flatMean, flatWidth, (shapeX, shapeY)) + rng2.normal(0.0, readNoise,
                                                                                 (shapeX, shapeY))
    # Simulate BF with power law model in galsim
    if len(powerLawBfParams):
        if not len(powerLawBfParams) == 8:
            raise RuntimeError("Wrong number of parameters for `galsim.cdmodel.PowerLawCD`. "
                               f"Expected 8; passed {len(powerLawBfParams)}.")
        cd = galsim.cdmodel.PowerLawCD(*powerLawBfParams)
        tempFlatData1 = galsim.Image(flatData1)
        temp2FlatData1 = cd.applyForward(tempFlatData1)

        tempFlatData2 = galsim.Image(flatData2)
        temp2FlatData2 = cd.applyForward(tempFlatData2)

        flatExp1.image.array[:] = temp2FlatData1.array/gain   # ADU
        flatExp2.image.array[:] = temp2FlatData2.array/gain  # ADU
    else:
        flatExp1.image.array[:] = flatData1/gain   # ADU
        flatExp2.image.array[:] = flatData2/gain   # ADU

    visitInfoExp1 = lsst.afw.image.VisitInfo(exposureId=expId1, exposureTime=expTime)
    visitInfoExp2 = lsst.afw.image.VisitInfo(exposureId=expId2, exposureTime=expTime)

    flatExp1.getInfo().setVisitInfo(visitInfoExp1)
    flatExp2.getInfo().setVisitInfo(visitInfoExp2)

    return flatExp1, flatExp2


def countMaskedPixels(maskedIm, maskPlane):
    """Count the number of pixels in a given mask plane."""
    maskBit = maskedIm.mask.getPlaneBitMask(maskPlane)
    nPix = np.where(np.bitwise_and(maskedIm.mask.array, maskBit))[0].flatten().size
    return nPix


class PairedVisitListTaskRunner(pipeBase.TaskRunner):
    """Subclass of TaskRunner for handling intrinsically paired visits.

    This transforms the processed arguments generated by the ArgumentParser
    into the arguments expected by tasks which take visit pairs for their
    run() methods.

    Such tasks' run() methods tend to take two arguments,
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


def parseCmdlineNumberString(inputString):
    """Parse command line numerical expression sytax and return as list of int

    Take an input of the form "'1..5:2^123..126'" as a string, and return
    a list of ints as [1, 3, 5, 123, 124, 125, 126]
    """
    outList = []
    for subString in inputString.split("^"):
        mat = re.search(r"^(\d+)\.\.(\d+)(?::(\d+))?$", subString)
        if mat:
            v1 = int(mat.group(1))
            v2 = int(mat.group(2))
            v3 = mat.group(3)
            v3 = int(v3) if v3 else 1
            for v in range(v1, v2 + 1, v3):
                outList.append(int(v))
        else:
            outList.append(int(subString))
    return outList


class SingleVisitListTaskRunner(pipeBase.TaskRunner):
    """Subclass of TaskRunner for tasks requiring a list of visits per dataRef.

    This transforms the processed arguments generated by the ArgumentParser
    into the arguments expected by tasks which require a list of visits
    to be supplied for each dataRef, as is common in `lsst.cp.pipe` code.

    Such tasks' run() methods tend to take two arguments,
    one of which is the dataRef (as usual), and the other is the list
    of visits.
    This list is supplied on the command line as documented,
    and this class parses that, and passes the parsed version
    to the run() method.

    See `lsst.pipe.base.TaskRunner` for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Parse the visit list and pass through explicitly."""
        # if this has been pre-parsed and therefore doesn't have length of one
        # then something has gone wrong, so execution should stop here.
        assert len(parsedCmd.visitList) == 1, 'visitList parsing assumptions violated'
        visits = parseCmdlineNumberString(parsedCmd.visitList[0])

        return pipeBase.TaskRunner.getTargetList(parsedCmd, visitList=visits, **kwargs)


class NonexistentDatasetTaskDataIdContainer(pipeBase.DataIdContainer):
    """A DataIdContainer for the tasks for which the output does
    not yet exist."""

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
        Note that this is almost a copy-and-paste of the vanilla
        implementation, but without checking if the datasets already exist,
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


def irlsFit(initialParams, dataX, dataY, function, weightsY=None):
    """Iteratively reweighted least squares fit.

    This uses the `lsst.cp.pipe.utils.fitLeastSq`, but applies
    weights based on the Cauchy distribution to the fitter.  See
    e.g. Holland and Welsch, 1977, doi:10.1080/03610927708827533

    Parameters
    ----------
    initialParams : `list` [`float`]
        Starting parameters.
    dataX : `numpy.array` [`float`]
        Abscissa data.
    dataY : `numpy.array` [`float`]
        Ordinate data.
    function : callable
        Function to fit.
    weightsY : `numpy.array` [`float`]
        Weights to apply to the data.

    Returns
    -------
    polyFit : `list` [`float`]
        Final best fit parameters.
    polyFitErr : `list` [`float`]
        Final errors on fit parameters.
    chiSq : `float`
        Reduced chi squared.
    weightsY : `list` [`float`]
        Final weights used for each point.

    """
    if not weightsY:
        weightsY = np.ones_like(dataX)

    polyFit, polyFitErr, chiSq = fitLeastSq(initialParams, dataX, dataY, function, weightsY=weightsY)
    for iteration in range(10):
        # Use Cauchy weights
        resid = np.abs(dataY - function(polyFit, dataX)) / np.sqrt(dataY)
        weightsY = 1.0 / (1.0 + np.sqrt(resid / 2.385))
        polyFit, polyFitErr, chiSq = fitLeastSq(initialParams, dataX, dataY, function, weightsY=weightsY)

    return polyFit, polyFitErr, chiSq, weightsY


def fitLeastSq(initialParams, dataX, dataY, function, weightsY=None):
    """Do a fit and estimate the parameter errors using using scipy.optimize.leastq.

    optimize.leastsq returns the fractional covariance matrix. To estimate the
    standard deviation of the fit parameters, multiply the entries of this matrix
    by the unweighted reduced chi squared and take the square root of the diagonal elements.

    Parameters
    ----------
    initialParams : `list` of `float`
        initial values for fit parameters. For ptcFitType=POLYNOMIAL, its length
        determines the degree of the polynomial.

    dataX : `numpy.array` of `float`
        Data in the abscissa axis.

    dataY : `numpy.array` of `float`
        Data in the ordinate axis.

    function : callable object (function)
        Function to fit the data with.

    weightsY : `numpy.array` of `float`
        Weights of the data in the ordinate axis.

    Return
    ------
    pFitSingleLeastSquares : `list` of `float`
        List with fitted parameters.

    pErrSingleLeastSquares : `list` of `float`
        List with errors for fitted parameters.

    reducedChiSqSingleLeastSquares : `float`
        Reduced chi squared, unweighted if weightsY is not provided.
    """
    if weightsY is None:
        weightsY = np.ones(len(dataX))

    def errFunc(p, x, y, weightsY=None):
        if weightsY is None:
            weightsY = np.ones(len(x))
        return (function(p, x) - y)*weightsY

    pFit, pCov, infoDict, errMessage, success = leastsq(errFunc, initialParams,
                                                        args=(dataX, dataY, weightsY), full_output=1,
                                                        epsfcn=0.0001)

    if (len(dataY) > len(initialParams)) and pCov is not None:
        reducedChiSq = calculateWeightedReducedChi2(dataY, function(pFit, dataX), weightsY, len(dataY),
                                                    len(initialParams))
        pCov *= reducedChiSq
    else:
        pCov = np.zeros((len(initialParams), len(initialParams)))
        pCov[:, :] = np.nan
        reducedChiSq = np.nan

    errorVec = []
    for i in range(len(pFit)):
        errorVec.append(np.fabs(pCov[i][i])**0.5)

    pFitSingleLeastSquares = pFit
    pErrSingleLeastSquares = np.array(errorVec)

    return pFitSingleLeastSquares, pErrSingleLeastSquares, reducedChiSq


def fitBootstrap(initialParams, dataX, dataY, function, weightsY=None, confidenceSigma=1.):
    """Do a fit using least squares and bootstrap to estimate parameter errors.

    The bootstrap error bars are calculated by fitting 100 random data sets.

    Parameters
    ----------
    initialParams : `list` of `float`
        initial values for fit parameters. For ptcFitType=POLYNOMIAL, its length
        determines the degree of the polynomial.

    dataX : `numpy.array` of `float`
        Data in the abscissa axis.

    dataY : `numpy.array` of `float`
        Data in the ordinate axis.

    function : callable object (function)
        Function to fit the data with.

    weightsY : `numpy.array` of `float`, optional.
        Weights of the data in the ordinate axis.

    confidenceSigma : `float`, optional.
        Number of sigmas that determine confidence interval for the bootstrap errors.

    Return
    ------
    pFitBootstrap : `list` of `float`
        List with fitted parameters.

    pErrBootstrap : `list` of `float`
        List with errors for fitted parameters.

    reducedChiSqBootstrap : `float`
        Reduced chi squared, unweighted if weightsY is not provided.
    """
    if weightsY is None:
        weightsY = np.ones(len(dataX))

    def errFunc(p, x, y, weightsY):
        if weightsY is None:
            weightsY = np.ones(len(x))
        return (function(p, x) - y)*weightsY

    # Fit first time
    pFit, _ = leastsq(errFunc, initialParams, args=(dataX, dataY, weightsY), full_output=0)

    # Get the stdev of the residuals
    residuals = errFunc(pFit, dataX, dataY, weightsY)
    # 100 random data sets are generated and fitted
    pars = []
    for i in range(100):
        randomDelta = np.random.normal(0., np.fabs(residuals), len(dataY))
        randomDataY = dataY + randomDelta
        randomFit, _ = leastsq(errFunc, initialParams,
                               args=(dataX, randomDataY, weightsY), full_output=0)
        pars.append(randomFit)
    pars = np.array(pars)
    meanPfit = np.mean(pars, 0)

    # confidence interval for parameter estimates
    errPfit = confidenceSigma*np.std(pars, 0)
    pFitBootstrap = meanPfit
    pErrBootstrap = errPfit

    reducedChiSq = calculateWeightedReducedChi2(dataY, function(pFitBootstrap, dataX), weightsY, len(dataY),
                                                len(initialParams))
    return pFitBootstrap, pErrBootstrap, reducedChiSq


def funcPolynomial(pars, x):
    """Polynomial function definition
    Parameters
    ----------
    params : `list`
        Polynomial coefficients. Its length determines the polynomial order.

    x : `numpy.array`
        Abscisa array.

    Returns
    -------
    Ordinate array after evaluating polynomial of order len(pars)-1 at `x`.
    """
    return poly.polyval(x, [*pars])


def funcAstier(pars, x):
    """Single brighter-fatter parameter model for PTC; Equation 16 of Astier+19.

    Parameters
    ----------
    params : `list`
        Parameters of the model: a00 (brightter-fatter), gain (e/ADU), and noise (e^2).

    x : `numpy.array`
        Signal mu (ADU).

    Returns
    -------
    C_00 (variance) in ADU^2.
    """
    a00, gain, noise = pars
    return 0.5/(a00*gain*gain)*(np.exp(2*a00*x*gain)-1) + noise/(gain*gain)  # C_00


def arrangeFlatsByExpTime(exposureList):
    """Arrange exposures by exposure time.

    Parameters
    ----------
    exposureList : `list`[`lsst.afw.image.exposure.exposure.ExposureF`]
        Input list of exposures.

    Returns
    ------
    flatsAtExpTime : `dict` [`float`,
                      `list`[`lsst.afw.image.exposure.exposure.ExposureF`]]
        Dictionary that groups flat-field exposures that have the same
        exposure time (seconds).
    """
    flatsAtExpTime = {}
    for exp in exposureList:
        tempFlat = exp
        expTime = tempFlat.getInfo().getVisitInfo().getExposureTime()
        listAtExpTime = flatsAtExpTime.setdefault(expTime, [])
        listAtExpTime.append(tempFlat)

    return flatsAtExpTime


def arrangeFlatsByExpId(exposureList):
    """Arrange exposures by exposure ID.

    Parameters
    ----------
    exposureList : `list`[`lsst.afw.image.exposure.exposure.ExposureF`]
        Input list of exposures.

    Returns
    ------
    flatsAtExpTime : `dict` [`float`,
                      `list`[`lsst.afw.image.exposure.exposure.ExposureF`]]
        Dictionary that groups flat-field exposures that have the same
        exposure time (seconds).
    """
    flatsAtExpId = {}
    sortedExposures = sorted(exposureList, key=lambda exp: exp.getInfo().getVisitInfo().getExposureId())

    for jPair, exp in enumerate(sortedExposures):
        if (jPair + 1) % 2:
            kPair = jPair // 2
            listAtExpId = flatsAtExpId.setdefault(kPair, [])
            listAtExpId.append(exp)
            try:
                listAtExpId.append(sortedExposures[jPair + 1])
            except IndexError:
                pass

    return flatsAtExpId


def checkExpLengthEqual(exp1, exp2, v1=None, v2=None, raiseWithMessage=False):
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
    raiseWithMessage : `bool`
        If True, instead of returning a bool, raise a RuntimeError if exposure
    times are not equal, with a message about which visits mismatch if the
    information is available.

    Raises:
    -------
    RuntimeError
        Raised if the exposure lengths of the two exposures are not equal
    """
    expTime1 = exp1.getInfo().getVisitInfo().getExposureTime()
    expTime2 = exp2.getInfo().getVisitInfo().getExposureTime()
    if expTime1 != expTime2:
        if raiseWithMessage:
            msg = "Exposure lengths for visit pairs must be equal. " + \
                  "Found %s and %s" % (expTime1, expTime2)
            if v1 and v2:
                msg += " for visit pair %s, %s" % (v1, v2)
            raise RuntimeError(msg)
        else:
            return False
    return True


def validateIsrConfig(isrTask, mandatory=None, forbidden=None, desirable=None, undesirable=None,
                      checkTrim=True, logName=None):
    """Check that appropriate ISR settings have been selected for the task.

    Note that this checks that the task itself is configured correctly rather
    than checking a config.

    Parameters
    ----------
    isrTask : `lsst.ip.isr.IsrTask`
        The task whose config is to be validated

    mandatory : `iterable` of `str`
        isr steps that must be set to True. Raises if False or missing

    forbidden : `iterable` of `str`
        isr steps that must be set to False. Raises if True, warns if missing

    desirable : `iterable` of `str`
        isr steps that should probably be set to True. Warns is False, info if
    missing

    undesirable : `iterable` of `str`
        isr steps that should probably be set to False. Warns is True, info if
    missing

    checkTrim : `bool`
        Check to ensure the isrTask's assembly subtask is trimming the images.
    This is a separate config as it is very ugly to do this within the
    normal configuration lists as it is an option of a sub task.

    Raises
    ------
    RuntimeError
        Raised if ``mandatory`` config parameters are False,
        or if ``forbidden`` parameters are True.

    TypeError
        Raised if parameter ``isrTask`` is an invalid type.

    Notes
    -----
    Logs warnings using an isrValidation logger for desirable/undesirable
    options that are of the wrong polarity or if keys are missing.
    """
    if not isinstance(isrTask, ipIsr.IsrTask):
        raise TypeError(f'Must supply an instance of lsst.ip.isr.IsrTask not {type(isrTask)}')

    configDict = isrTask.config.toDict()

    if logName and isinstance(logName, str):
        log = lsst.log.getLogger(logName)
    else:
        log = lsst.log.getLogger("isrValidation")

    if mandatory:
        for configParam in mandatory:
            if configParam not in configDict:
                raise RuntimeError(f"Mandatory parameter {configParam} not found in the isr configuration.")
            if configDict[configParam] is False:
                raise RuntimeError(f"Must set config.isr.{configParam} to True for this task.")

    if forbidden:
        for configParam in forbidden:
            if configParam not in configDict:
                log.warn(f"Failed to find forbidden key {configParam} in the isr config. The keys in the"
                         " forbidden list should each have an associated Field in IsrConfig:"
                         " check that there is not a typo in this case.")
                continue
            if configDict[configParam] is True:
                raise RuntimeError(f"Must set config.isr.{configParam} to False for this task.")

    if desirable:
        for configParam in desirable:
            if configParam not in configDict:
                log.info(f"Failed to find key {configParam} in the isr config. You probably want"
                         " to set the equivalent for your obs_package to True.")
                continue
            if configDict[configParam] is False:
                log.warn(f"Found config.isr.{configParam} set to False for this task."
                         " The cp_pipe Config recommends setting this to True.")
    if undesirable:
        for configParam in undesirable:
            if configParam not in configDict:
                log.info(f"Failed to find key {configParam} in the isr config. You probably want"
                         " to set the equivalent for your obs_package to False.")
                continue
            if configDict[configParam] is True:
                log.warn(f"Found config.isr.{configParam} set to True for this task."
                         " The cp_pipe Config recommends setting this to False.")

    if checkTrim:  # subtask setting, seems non-trivial to combine with above lists
        if not isrTask.assembleCcd.config.doTrim:
            raise RuntimeError("Must trim when assembling CCDs. Set config.isr.assembleCcd.doTrim to True")


def ddict2dict(d):
    """Convert nested default dictionaries to regular dictionaries.

    This is needed to prevent yaml persistence issues.

    Parameters
    ----------
    d : `defaultdict`
        A possibly nested set of `defaultdict`.

    Returns
    -------
    dict : `dict`
        A possibly nested set of `dict`.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)
