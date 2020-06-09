#!/usr/bin/env python

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
"""Test cases for cp_pipe."""

from __future__ import absolute_import, division, print_function
import unittest
import numpy as np
import copy

import lsst.utils
import lsst.utils.tests

import lsst.cp.pipe as cpPipe
import lsst.ip.isr.isrMock as isrMock
from lsst.cp.pipe.ptc import PhotonTransferCurveDataset
from lsst.cp.pipe.astierCovPtcUtils import fitData
import lsst.afw.math as afwMath


class MeasurePhotonTransferCurveTaskTestCase(lsst.utils.tests.TestCase):
    """A test case for the PTC task."""

    def setUp(self):
        self.defaultConfig = cpPipe.ptc.MeasurePhotonTransferCurveTask.ConfigClass()
        self.defaultConfig.isr.doFlat = False
        self.defaultConfig.isr.doFringe = False
        self.defaultConfig.isr.doCrosstalk = False
        self.defaultConfig.isr.doUseOpticsTransmission = False
        self.defaultConfig.isr.doUseFilterTransmission = False
        self.defaultConfig.isr.doUseSensorTransmission = False
        self.defaultConfig.isr.doUseAtmosphereTransmission = False
        self.defaultConfig.isr.doAttachTransmissionCurve = False

        self.defaultTask = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=self.defaultConfig)

        self.flatMean = 2000
        self.readNoiseAdu = 10
        mockImageConfig = isrMock.IsrMock.ConfigClass()

        # flatDrop is not really relevant as we replace the data
        # but good to note it in case we change how this image is made
        mockImageConfig.flatDrop = 0.99999
        mockImageConfig.isTrimmed = True

        self.flatExp1 = isrMock.FlatMock(config=mockImageConfig).run()
        self.flatExp2 = self.flatExp1.clone()
        (shapeY, shapeX) = self.flatExp1.getDimensions()

        self.flatWidth = np.sqrt(self.flatMean) + self.readNoiseAdu

        self.rng1 = np.random.RandomState(1984)
        flatData1 = self.rng1.normal(self.flatMean, self.flatWidth, (shapeX, shapeY))
        self.rng2 = np.random.RandomState(666)
        flatData2 = self.rng2.normal(self.flatMean, self.flatWidth, (shapeX, shapeY))

        self.flatExp1.image.array[:] = flatData1
        self.flatExp2.image.array[:] = flatData2

        # create fake PTC data to see if fit works, for one amp ('amp')
        self.flux = 1000.  # ADU/sec
        timeVec = np.arange(1., 201.)
        self.k2NonLinearity = -5e-6
        muVec = self.flux*timeVec + self.k2NonLinearity*timeVec**2   # quadratic signal-chain non-linearity
        self.gain = 1.5  # e-/ADU
        self.c1 = 1./self.gain
        self.noiseSq = 5*self.gain  # 7.5 (e-)^2
        self.a00 = -1.2e-6
        self.c2 = -1.5e-6
        self.c3 = -4.7e-12  # tuned so that it turns over for 200k mean

        self.ampNames = [amp.getName() for amp in self.flatExp1.getDetector().getAmplifiers()]
        self.dataset = PhotonTransferCurveDataset(self.ampNames)  # pack raw data for fitting

        for ampName in self.ampNames:  # just the expTimes and means here - vars vary per function
            self.dataset.rawExpTimes[ampName] = timeVec
            self.dataset.rawMeans[ampName] = muVec

    def makeMockFlats(self, expTime, gain=1.0, readNoiseElectrons=5, fluxElectrons=1000):
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

        rng1 = np.random.RandomState(1984)
        flatData1 = (rng1.normal(flatMean, flatWidth, (shapeX, shapeY)) +
                     rng1.normal(0.0, readNoise, (shapeX, shapeY)))
        rng2 = np.random.RandomState(666)
        flatData2 = (rng2.normal(flatMean, flatWidth, (shapeX, shapeY)) +
                     rng2.normal(0.0, readNoise, (shapeX, shapeY)))

        flatExp1.image.array[:] = flatData1/gain  # ADU
        flatExp2.image.array[:] = flatData2/gain  # ADU

        return flatExp1, flatExp2

    def test_covAstier(self):
        """Test to check getCovariancesAstier

        We check that the gain is the same as the imput gain from the mock data, that
        the covariances via FFT (as it is in MeasurePhotonTransferCurveTask when
        doCovariancesAstier=True) are the same as calculated in real space, and that
        Cov[0, 0] (i.e., the variances) are similar to the variances calculated with the standard
        method (when doCovariancesAstier=false),
        """
        config = copy.copy(self.defaultConfig)
        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)

        expTimes = np.arange(5, 170, 5)
        tupleRecords = []
        allTags = []
        muStandard, varStandard = {}, {}
        for expTime in expTimes:
            mockExp1, mockExp2 = self.makeMockFlats(expTime, gain=0.75)
            tupleRows = []

            for ampNumber, amp in enumerate(self.ampNames):
                # cov has (i, j, var, cov, npix)
                mu1, mu2, covs = task.getCovariancesAstier(mockExp1, mockExp2)
                # calculate mean and variance of difference image in the standard way
                mu, varDiff = task.measureMeanVarPair(mockExp1, mockExp2)
                muStandard.setdefault(amp, []).append(mu)
                varStandard.setdefault(amp, []).append(varDiff)
                # Calculate covariances in an independent way: direct space
                _, _, covsDirect = self.getCovariancesAstierDirect(mockExp1, mockExp2, config)
                # Test that the arrays "covs" (FFT) and "covDirect" (direct space) are the same
                for row1, row2 in zip(covs, covsDirect):
                    for a, b in zip(row1, row2):
                        self.assertAlmostEqual(a, b)
                tupleRows += [(mu1, mu2) + cov + (ampNumber, expTime, amp) for cov in covs]
                tags = ['mu1', 'mu2', 'i', 'j', 'var', 'cov', 'npix', 'ext', 'expTime', 'ampName']
            allTags += tags
            tupleRecords += tupleRows
        covariancesWithTags = np.core.records.fromrecords(tupleRecords, names=allTags)
        covFits, _ = fitData(covariancesWithTags)
        dataset = task.getOutputPtcDataCovAstier(covFits)

        # Chek the gain and that the ratio of the variance caclulated via cov Astier (FFT) and
        # that calculated with the standard PTC is close to 1.
        for amp in self.ampNames:
            self.assertAlmostEqual(dataset.gain[amp][0], 0.75, places=2)
            for v1, v2 in zip(varStandard[amp], dataset.varAdu[amp][0]):
                self.assertAlmostEqual(v1/v2, 1.0, places=1)

    def getCovariancesAstierDirect(self, exposure1, exposure2, config, region=None):
        """Calculate covariances of a difference image in real space"""
        if region is not None:
            im1Area = exposure1.maskedImage[region]
            im2Area = exposure2.maskedImage[region]
        else:
            im1Area = exposure1.maskedImage
            im2Area = exposure2.maskedImage

        im1Area = afwMath.binImage(im1Area, config.binSize)
        im2Area = afwMath.binImage(im2Area, config.binSize)

        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(config.nSigmaClipPtc)
        statsCtrl.setNumIter(config.nIterSigmaClipPtc)
        #  Clipped mean of images; then average of mean.
        mu1 = afwMath.makeStatistics(im1Area, afwMath.MEANCLIP, statsCtrl).getValue()
        mu2 = afwMath.makeStatistics(im2Area, afwMath.MEANCLIP, statsCtrl).getValue()
        mu = 0.5*(mu1 + mu2)

        # Take difference of pairs
        # symmetric formula: diff = (mu2*im1-mu1*im2)/(0.5*(mu1+mu2))
        temp = im2Area.clone()
        temp *= mu1
        diffIm = im1Area.clone()
        diffIm *= mu2
        diffIm -= temp
        diffIm /= mu

        # Get the mask and identify good pixels as '1', and the rest as '0'.
        w1 = np.where(im1Area.getMask().getArray() == 0, 1, 0)
        w2 = np.where(im2Area.getMask().getArray() == 0, 1, 0)

        w12 = w1*w2
        wDiff = np.where(diffIm.getMask().getArray() == 0, 1, 0)
        w = w12*wDiff

        maxRangeCov = config.maximumRangeCovariancesAstier
        covs = self.computeCovDirect(diffIm.getImage().getArray(), w, maxRangeCov)

        return mu1, mu2, covs

    def computeCovDirect(self, diffImage, weightImage, maxRange):
        """Compute covariances of diffImage in real space.

        For lags larger than ~25, it is slower than the FFT way.
        Taken from https://github.com/PierreAstier/bfptc/

        Parameters
        ----------
        diffImage : `numpy.array`
            Image to compute the covariance of.

        weightImage : `numpy.array`
            Weight image of diffImage (1's and 0's for good and bad pixels, respectively).

        maxRange : `int`
            Last index of the covariance to be computed.

        Returns
        -------
        outList : `list`
            List with tuples of the form (dx, dy, var, cov, npix), where:
            dx : `int`
                Lag in x
            dy : `int`
                Lag in y
            var : `float`
                Variance at (dx, dy).
            cov : `float`
                Covariance at (dx, dy).
            nPix : `int`
                Number of pixel pairs used to evaluate var and cov.
        """
        outList = []
        var = 0
        # (dy,dx) = (0,0) has to be first
        for dy in range(maxRange + 1):
            for dx in range(0, maxRange + 1):
                if (dx*dy > 0):
                    cov1, nPix1 = self.covDirectValue(diffImage, weightImage, dx, dy)
                    cov2, nPix2 = self.covDirectValue(diffImage, weightImage, dx, -dy)
                    cov = 0.5*(cov1 + cov2)
                    nPix = nPix1 + nPix2
                else:
                    cov, nPix = self.covDirectValue(diffImage, weightImage, dx, dy)
                if (dx == 0 and dy == 0):
                    var = cov
                outList.append((dx, dy, var, cov, nPix))

        return outList

    def covDirectValue(self, diffImage, weightImage, dx, dy):
        """Compute covariances of diffImage in real space at lag (dx, dy).

        Taken from https://github.com/PierreAstier/bfptc/ (c.f., appendix of Astier+19).

        Parameters
        ----------
        diffImage : `numpy.array`
            Image to compute the covariance of.

        weightImage : `numpy.array`
            Weight image of diffImage (1's and 0's for good and bad pixels, respectively).

        dx : `int`
            Lag in x.

        dy : `int`
            Lag in y.

        Returns
        -------
        cov : `float`
            Covariance at (dx, dy)

        nPix : `int`
            Number of pixel pairs used to evaluate var and cov.
        """
        (nCols, nRows) = diffImage.shape
        # switching both signs does not change anything:
        # it just swaps im1 and im2 below
        if (dx < 0):
            (dx, dy) = (-dx, -dy)
        # now, we have dx >0. We have to distinguish two cases
        # depending on the sign of dy
        if dy >= 0:
            im1 = diffImage[dy:, dx:]
            w1 = weightImage[dy:, dx:]
            im2 = diffImage[:nCols - dy, :nRows - dx]
            w2 = weightImage[:nCols - dy, :nRows - dx]
        else:
            im1 = diffImage[:nCols + dy, dx:]
            w1 = weightImage[:nCols + dy, dx:]
            im2 = diffImage[-dy:, :nRows - dx]
            w2 = weightImage[-dy:, :nRows - dx]
        # use the same mask for all 3 calculations
        wAll = w1*w2
        # do not use mean() because weightImage=0 pixels would then count
        nPix = wAll.sum()
        im1TimesW = im1*wAll
        s1 = im1TimesW.sum()/nPix
        s2 = (im2*wAll).sum()/nPix
        p = (im1TimesW*im2).sum()/nPix
        cov = p - s1*s2

        return cov, nPix

    def ptcFitAndCheckPtc(self, order=None, fitType='', doTableArray=False):
        localDataset = copy.copy(self.dataset)
        config = copy.copy(self.defaultConfig)
        if fitType == 'POLYNOMIAL':
            if order not in [2, 3]:
                RuntimeError("Enter a valid polynomial order for this test: 2 or 3")
            if order == 2:
                for ampName in self.ampNames:
                    localDataset.rawVars[ampName] = [self.noiseSq + self.c1*mu + self.c2*mu**2 for
                                                     mu in localDataset.rawMeans[ampName]]
                config.polynomialFitDegree = 2
            if order == 3:
                for ampName in self.ampNames:
                    localDataset.rawVars[ampName] = [self.noiseSq + self.c1*mu + self.c2*mu**2 + self.c3*mu**3
                                                     for mu in localDataset.rawMeans[ampName]]
                config.polynomialFitDegree = 3
        elif fitType == 'ASTIERAPPROXIMATION':
            g = self.gain
            for ampName in self.ampNames:
                localDataset.rawVars[ampName] = [(0.5/(self.a00*g**2)*(np.exp(2*self.a00*mu*g)-1) +
                                                 self.noiseSq/(g*g)) for mu in localDataset.rawMeans[ampName]]
        else:
            RuntimeError("Enter a fit function type: 'POLYNOMIAL' or 'ASTIERAPPROXIMATION'")

        config.maxAduForLookupTableLinearizer = 200000  # Max ADU in input mock flats
        task = cpPipe.ptc.MeasurePhotonTransferCurveTask(config=config)

        if doTableArray:
            numberAmps = len(self.ampNames)
            numberAduValues = config.maxAduForLookupTableLinearizer
            lookupTableArray = np.zeros((numberAmps, numberAduValues), dtype=np.float32)
            returnedDataset = task.fitPtcAndNonLinearity(localDataset, ptcFitType=fitType,
                                                         tableArray=lookupTableArray)
        else:
            returnedDataset = task.fitPtcAndNonLinearity(localDataset, ptcFitType=fitType)

        if doTableArray:
            # check that the linearizer table has been filled out properly
            for i in np.arange(numberAmps):
                tMax = (config.maxAduForLookupTableLinearizer)/self.flux
                timeRange = np.linspace(0., tMax, config.maxAduForLookupTableLinearizer)
                signalIdeal = timeRange*self.flux
                signalUncorrected = task.funcPolynomial(np.array([0.0, self.flux, self.k2NonLinearity]),
                                                        timeRange)
                linearizerTableRow = signalIdeal - signalUncorrected
                self.assertEqual(len(linearizerTableRow), len(lookupTableArray[i, :]))
                for j in np.arange(len(linearizerTableRow)):
                    self.assertAlmostEqual(linearizerTableRow[j], lookupTableArray[i, :][j], places=6)

        # check entries in localDataset, which was modified by the function
        for ampName in self.ampNames:
            maskAmp = localDataset.visitMask[ampName]
            finalMuVec = localDataset.rawMeans[ampName][maskAmp]
            finalTimeVec = localDataset.rawExpTimes[ampName][maskAmp]
            inputNonLinearityResiduals = 100*(1 - ((finalMuVec[2]/finalTimeVec[2])/(finalMuVec/finalTimeVec)))
            linearPart = self.flux*finalTimeVec
            inputFracNonLinearityResiduals = 100*(linearPart - finalMuVec)/linearPart

            self.assertEqual(fitType, localDataset.ptcFitType[ampName])
            self.assertAlmostEqual(self.gain, localDataset.gain[ampName])
            if fitType == 'POLYNOMIAL':
                self.assertAlmostEqual(self.c1, localDataset.ptcFitPars[ampName][1])
                self.assertAlmostEqual(np.sqrt(self.noiseSq)*self.gain, localDataset.noise[ampName])
            else:
                self.assertAlmostEqual(self.a00, localDataset.ptcFitPars[ampName][0])
                # noise already in electrons for 'ASTIERAPPROXIMATION' fit
                self.assertAlmostEqual(np.sqrt(self.noiseSq), localDataset.noise[ampName])
            # Nonlinearity fit parameters
            self.assertAlmostEqual(0.0, localDataset.nonLinearity[ampName][0])
            self.assertAlmostEqual(self.flux, localDataset.nonLinearity[ampName][1])
            self.assertAlmostEqual(self.k2NonLinearity, localDataset.nonLinearity[ampName][2])

            # Non-linearity coefficient for quadratic linearizer
            self.assertAlmostEqual(-self.k2NonLinearity/(self.flux**2),
                                   localDataset.coefficientLinearizeSquared[ampName])

            # Linearity residuals
            self.assertEqual(len(localDataset.nonLinearityResiduals[ampName]),
                             len(inputNonLinearityResiduals))
            for calc, truth in zip(localDataset.nonLinearityResiduals[ampName],
                                   inputNonLinearityResiduals):
                self.assertAlmostEqual(calc, truth)

            # Fractional nonlinearity residuals
            self.assertEqual(len(localDataset.fractionalNonLinearityResiduals[ampName]),
                             len(inputFracNonLinearityResiduals))
            for calc, truth in zip(localDataset.fractionalNonLinearityResiduals[ampName],
                                   inputFracNonLinearityResiduals):
                self.assertAlmostEqual(calc, truth)

            # check calls to calculateLinearityResidualAndLinearizers
            datasetLinResAndLinearizers = task.calculateLinearityResidualAndLinearizers(
                localDataset.rawExpTimes[ampName], localDataset.rawMeans[ampName])

            self.assertAlmostEqual(-self.k2NonLinearity/(self.flux**2),
                                   datasetLinResAndLinearizers.quadraticPolynomialLinearizerCoefficient)
            self.assertAlmostEqual(0.0, datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[0])
            self.assertAlmostEqual(self.flux, datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[1])
            self.assertAlmostEqual(self.k2NonLinearity,
                                   datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[2])

        # check entries in returned dataset (should be the same as localDataset after calling the function)
        for ampName in self.ampNames:
            maskAmp = returnedDataset.visitMask[ampName]
            finalMuVec = returnedDataset.rawMeans[ampName][maskAmp]
            finalTimeVec = returnedDataset.rawExpTimes[ampName][maskAmp]
            inputNonLinearityResiduals = 100*(1 - ((finalMuVec[2]/finalTimeVec[2])/(finalMuVec/finalTimeVec)))
            linearPart = self.flux*finalTimeVec
            inputFracNonLinearityResiduals = 100*(linearPart - finalMuVec)/linearPart

            self.assertEqual(fitType, returnedDataset.ptcFitType[ampName])
            self.assertAlmostEqual(self.gain, returnedDataset.gain[ampName])
            if fitType == 'POLYNOMIAL':
                self.assertAlmostEqual(self.c1, returnedDataset.ptcFitPars[ampName][1])
                self.assertAlmostEqual(np.sqrt(self.noiseSq)*self.gain, returnedDataset.noise[ampName])
            else:
                self.assertAlmostEqual(self.a00, returnedDataset.ptcFitPars[ampName][0])
                # noise already in electrons for 'ASTIERAPPROXIMATION' fit
                self.assertAlmostEqual(np.sqrt(self.noiseSq), returnedDataset.noise[ampName])

            # Nonlinearity fit parameters
            self.assertAlmostEqual(0.0, returnedDataset.nonLinearity[ampName][0])
            self.assertAlmostEqual(self.flux, returnedDataset.nonLinearity[ampName][1])
            self.assertAlmostEqual(self.k2NonLinearity, returnedDataset.nonLinearity[ampName][2])

            # Non-linearity coefficient for linearizer
            self.assertAlmostEqual(-self.k2NonLinearity/(self.flux**2),
                                   returnedDataset.coefficientLinearizeSquared[ampName])

            # Linearity residuals
            self.assertEqual(len(returnedDataset.nonLinearityResiduals[ampName]),
                             len(inputNonLinearityResiduals))
            for calc, truth in zip(returnedDataset.nonLinearityResiduals[ampName],
                                   inputNonLinearityResiduals):
                self.assertAlmostEqual(calc, truth)

            # Fractional nonlinearity residuals
            self.assertEqual(len(returnedDataset.fractionalNonLinearityResiduals[ampName]),
                             len(inputFracNonLinearityResiduals))
            for calc, truth in zip(returnedDataset.fractionalNonLinearityResiduals[ampName],
                                   inputFracNonLinearityResiduals):
                self.assertAlmostEqual(calc, truth)

            # check calls to calculateLinearityResidualAndLinearizers
            datasetLinResAndLinearizers = task.calculateLinearityResidualAndLinearizers(
                returnedDataset.rawExpTimes[ampName], returnedDataset.rawMeans[ampName])

            self.assertAlmostEqual(-self.k2NonLinearity/(self.flux**2),
                                   datasetLinResAndLinearizers.quadraticPolynomialLinearizerCoefficient)
            self.assertAlmostEqual(0.0, datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[0])
            self.assertAlmostEqual(self.flux, datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[1])
            self.assertAlmostEqual(self.k2NonLinearity,
                                   datasetLinResAndLinearizers.meanSignalVsTimePolyFitPars[2])

    def test_ptcFit(self):
        for createArray in [True, False]:
            for typeAndOrder in [('POLYNOMIAL', 2), ('POLYNOMIAL', 3), ('ASTIERAPPROXIMATION', None)]:
                self.ptcFitAndCheckPtc(fitType=typeAndOrder[0], order=typeAndOrder[1],
                                       doTableArray=createArray)

    def test_meanVarMeasurement(self):
        task = self.defaultTask
        mu, varDiff = task.measureMeanVarPair(self.flatExp1, self.flatExp2)

        self.assertLess(self.flatWidth - np.sqrt(varDiff), 1)
        self.assertLess(self.flatMean - mu, 1)

    def test_getInitialGoodPoints(self):
        xs = [1, 2, 3, 4, 5, 6]
        ys = [2*x for x in xs]
        points = self.defaultTask._getInitialGoodPoints(xs, ys, 0.1, 0.25)
        assert np.all(points) == np.all(np.array([True for x in xs]))

        ys[-1] = 30
        points = self.defaultTask._getInitialGoodPoints(xs, ys, 0.1, 0.25)
        assert np.all(points) == np.all(np.array([True, True, True, True, False]))

        ys = [2*x for x in xs]
        newYs = copy.copy(ys)
        results = [False, True, True, False, False]
        for i, factor in enumerate([-0.5, -0.1, 0, 0.1, 0.5]):
            newYs[-1] = ys[-1] + (factor*ys[-1])
            points = self.defaultTask._getInitialGoodPoints(xs, newYs, 0.05, 0.25)
            assert (np.all(points[0:-1]) == True)  # noqa: E712 - flake8 is wrong here because of numpy.bool
            assert points[-1] == results[i]

    def test_getVisitsUsed(self):
        localDataset = copy.copy(self.dataset)

        for pair in [(12, 34), (56, 78), (90, 10)]:
            localDataset.inputVisitPairs["C:0,0"].append(pair)
        localDataset.visitMask["C:0,0"] = np.array([True, False, True])
        self.assertTrue(np.all(localDataset.getVisitsUsed("C:0,0") == [(12, 34), (90, 10)]))

        localDataset.visitMask["C:0,0"] = np.array([True, False, True, True])  # wrong length now
        with self.assertRaises(AssertionError):
            localDataset.getVisitsUsed("C:0,0")

    def test_getGoodAmps(self):
        dataset = self.dataset

        self.assertTrue(dataset.ampNames == self.ampNames)
        dataset.badAmps.append("C:0,1")
        self.assertTrue(dataset.getGoodAmps() == [amp for amp in self.ampNames if amp != "C:0,1"])


class MeasurePhotonTransferCurveDatasetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.ptcData = PhotonTransferCurveDataset(['C00', 'C01'])
        self.ptcData.inputVisitPairs = {'C00': [(123, 234), (345, 456), (567, 678)],
                                        'C01': [(123, 234), (345, 456), (567, 678)]}

    def test_generalBehaviour(self):
        test = PhotonTransferCurveDataset(['C00', 'C01'])
        test.inputVisitPairs = {'C00': [(123, 234), (345, 456), (567, 678)],
                                'C01': [(123, 234), (345, 456), (567, 678)]}

        with self.assertRaises(AttributeError):
            test.newItem = 1


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
