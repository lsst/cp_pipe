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
import copy
import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from lsst.ip.isr import DeferredChargeCalib, SerialTrap
from lmfit import Minimizer, Parameters

from ._lookupStaticCalibration import lookupStaticCalibration

__all__ = ('CpCtiSolveConnections',
           'CpCtiSolveConfig',
           'CpCtiSolveTask',
           'OverscanModel',
           'SimpleModel',
           'SimulatedModel',
           'SegmentSimulator',
           'FloatingOutputAmplifier',
           )


class CpCtiSolveConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("instrument", "detector")):
    inputMeasurements = cT.Input(
        name="cpCtiMeas",
        doc="Input overscan measurements to fit.",
        storageClass='StructuredDataDict',
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera geometry to use.",
        storageClass="Camera",
        dimensions=("instrument", ),
        lookupFunction=lookupStaticCalibration,
        isCalibration=True,
    )

    outputCalib = cT.Output(
        name="cpCtiCalib",
        doc="Output CTI calibration.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
    )


class CpCtiSolveConfig(pipeBase.PipelineTaskConfig,
                       pipelineConnections=CpCtiSolveConnections):
    """Configuration for the CTI combination.
    """
    maxImageMean = pexConfig.Field(
        dtype=float,
        default=150000.0,
        doc="Upper limit on acceptable image flux mean.",
    )
    localOffsetColumnRange = pexConfig.ListField(
        dtype=int,
        default=[3, 13],
        doc="First and last overscan column to use for local offset effect.",
    )

    maxSignalForCti = pexConfig.Field(
        dtype=float,
        default=10000.0,
        doc="Upper flux limit to use for CTI fit.",
    )
    globalCtiColumnRange = pexConfig.ListField(
        dtype=int,
        default=[1, 2],
        doc="First and last overscan column to use for global CTI fit.",
    )

    trapColumnRange = pexConfig.ListField(
        dtype=int,
        default=[1, 20],
        doc="First and last overscan column to use for serial trap fit.",
    )

    fitError = pexConfig.Field(
        # This gives the error on the mean in a given column, and so
        # is expected to be $RN / sqrt(N_rows)$.
        dtype=float,
        default=7.0/np.sqrt(2000),
        doc="Error to use during parameter fitting.",
    )


class CpCtiSolveTask(pipeBase.PipelineTask,
                     pipeBase.CmdLineTask):
    """Combine CTI measurements to a final calibration.

    This task uses the extended pixel edge response (EPER) method as
    described by Snyder et al. 2021, Journal of Astronimcal
    Telescopes, Instruments, and Systems, 7,
    048002. doi:10.1117/1.JATIS.7.4.048002
    """

    ConfigClass = CpCtiSolveConfig
    _DefaultName = 'cpCtiSolve'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allowDebug = True

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        dimensions = [exp.dataId.byName() for exp in inputRefs.inputMeasurements]
        inputs['inputDims'] = dimensions

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputMeasurements, camera, inputDims):
        """Solve for charge transfer inefficiency from overscan measurements.

        Parameters
        ----------
        inputMeasurements : `list` [`dict`]
            List of overscan measurements from each input exposure.
            Each dictionary is nested within a top level 'CTI' key,
            with measurements organized by amplifier name, containing
            keys:

            ``"FIRST_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"OVERSCAN_COLUMNS"``
                List of overscan column indicies (`list` [`int`]).
            ``"OVERSCAN_VALUES"``
                List of overscan column means (`list` [`float`]).
        camera : `lsst.afw.cameraGeom.Camera`
            Camera geometry to use to find detectors.
        inputDims : `list` [`dict`]
            List of input dimensions from each input exposure.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Result struct containing:

            ``outputCalib``
                Final CTI calibration data
                (`lsst.ip.isr.DeferredChargeCalib`).

        Raises
        ------
        RuntimeError
            Raised if data from multiple detectors are passed in.
        """
        detectorSet = set([d['detector'] for d in inputDims])
        if len(detectorSet) != 1:
            raise RuntimeError("Inputs for too many detectors passed.")
        detectorId = detectorSet.pop()
        detector = camera[detectorId]

        # Initialize with detector.
        calib = DeferredChargeCalib(camera=camera, detector=detector)

        localCalib = self.solveLocalOffsets(inputMeasurements, calib, detector)

        globalCalib = self.solveGlobalCti(inputMeasurements, localCalib, detector)

        finalCalib = self.findTraps(inputMeasurements, globalCalib, detector)

        return pipeBase.Struct(
            outputCalib=finalCalib,
        )

    def solveLocalOffsets(self, inputMeasurements, calib, detector):
        """Solve for local (pixel-to-pixel) electronic offsets.

        This method fits for \tau_L, the local electronic offset decay
        time constant, and A_L, the local electronic offset constant
        of proportionality.

        Parameters
        ----------
        inputMeasurements : `list` [`dict`]
            List of overscan measurements from each input exposure.
            Each dictionary is nested within a top level 'CTI' key,
            with measurements organized by amplifier name, containing
            keys:

            ``"FIRST_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"OVERSCAN_COLUMNS"``
                List of overscan column indicies (`list` [`int`]).
            ``"OVERSCAN_VALUES"``
                List of overscan column means (`list` [`float`]).
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Calibration to populate with values.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object containing the geometry information for
            the amplifiers.

        Returns
        -------
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Populated calibration.

        Notes
        -----
        The original CTISIM code uses a data model in which the
        "overscan" consists of the standard serial overscan bbox with
        the values for the last imaging data column prepended to that
        list.  This version of the code keeps the overscan and imaging
        sections separate, and so a -1 offset is needed to ensure that
        the same columns are used for fitting between this code and
        CTISIM.  This offset removes that last imaging data column
        from the count.
        """
        # Range to fit.  These are in "camera" coordinates, and so
        # need to have the count for last image column removed.
        start, stop = self.config.localOffsetColumnRange
        start -= 1
        stop -= 1

        # Loop over amps/inputs, fitting those columns from
        # "non-saturated" inputs.
        for amp in detector.getAmplifiers():
            ampName = amp.getName()

            # Number of serial shifts.
            nCols = amp.getRawDataBBox().getWidth() + amp.getRawSerialPrescanBBox().getWidth()

            # The signal is the mean intensity of each input, and the
            # data are the overscan columns to fit.  For detectors
            # with non-zero CTI, the charge from the imaging region
            # leaks into the overscan region.
            signal = []
            data = []
            Nskipped = 0
            for exposureEntry in inputMeasurements:
                exposureDict = exposureEntry['CTI']
                if exposureDict[ampName]['IMAGE_MEAN'] < self.config.maxImageMean:
                    signal.append(exposureDict[ampName]['IMAGE_MEAN'])
                    data.append(exposureDict[ampName]['OVERSCAN_VALUES'][start:stop+1])
                else:
                    Nskipped += 1
            self.log.info(f"Skipped {Nskipped} exposures brighter than {self.config.maxImageMean}.")

            signal = np.array(signal)
            data = np.array(data)

            ind = signal.argsort()
            signal = signal[ind]
            data = data[ind]

            params = Parameters()
            params.add('ctiexp', value=-6, min=-7, max=-5, vary=False)
            params.add('trapsize', value=0.0, min=0.0, max=10., vary=False)
            params.add('scaling', value=0.08, min=0.0, max=1.0, vary=False)
            params.add('emissiontime', value=0.4, min=0.1, max=1.0, vary=False)
            params.add('driftscale', value=0.00022, min=0., max=0.001, vary=True)
            params.add('decaytime', value=2.4, min=0.1, max=4.0, vary=True)

            model = SimpleModel()
            minner = Minimizer(model.difference, params,
                               fcn_args=(signal, data, self.config.fitError, nCols),
                               fcn_kws={'start': start, 'stop': stop})
            result = minner.minimize()

            # Save results for the drift scale and decay time.
            if not result.success:
                self.log.warning("Electronics fitting failure for amplifier %s.", ampName)

            calib.globalCti[ampName] = 10**result.params['ctiexp']
            calib.driftScale[ampName] = result.params['driftscale'].value if result.success else 0.0
            calib.decayTime[ampName] = result.params['decaytime'].value if result.success else 2.4
            self.log.info("CTI Local Fit %s: cti: %g decayTime: %g driftScale %g",
                          ampName, calib.globalCti[ampName], calib.decayTime[ampName],
                          calib.driftScale[ampName])
        return calib

    def solveGlobalCti(self, inputMeasurements, calib, detector):
        """Solve for global CTI constant.

        This method solves for the mean global CTI, b.

        Parameters
        ----------
        inputMeasurements : `list` [`dict`]
            List of overscan measurements from each input exposure.
            Each dictionary is nested within a top level 'CTI' key,
            with measurements organized by amplifier name, containing
            keys:

            ``"FIRST_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"OVERSCAN_COLUMNS"``
                List of overscan column indicies (`list` [`int`]).
            ``"OVERSCAN_VALUES"``
                List of overscan column means (`list` [`float`]).
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Calibration to populate with values.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object containing the geometry information for
            the amplifiers.

        Returns
        -------
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Populated calibration.

        Notes
        -----
        The original CTISIM code uses a data model in which the
        "overscan" consists of the standard serial overscan bbox with
        the values for the last imaging data column prepended to that
        list.  This version of the code keeps the overscan and imaging
        sections separate, and so a -1 offset is needed to ensure that
        the same columns are used for fitting between this code and
        CTISIM.  This offset removes that last imaging data column
        from the count.
        """
        # Range to fit.  These are in "camera" coordinates, and so
        # need to have the count for last image column removed.
        start, stop = self.config.globalCtiColumnRange
        start -= 1
        stop -= 1

        # Loop over amps/inputs, fitting those columns from
        # "non-saturated" inputs.
        for amp in detector.getAmplifiers():
            ampName = amp.getName()

            # Number of serial shifts.
            nCols = amp.getRawDataBBox().getWidth() + amp.getRawSerialPrescanBBox().getWidth()

            # The signal is the mean intensity of each input, and the
            # data are the overscan columns to fit.  For detectors
            # with non-zero CTI, the charge from the imaging region
            # leaks into the overscan region.
            signal = []
            data = []
            Nskipped = 0
            for exposureEntry in inputMeasurements:
                exposureDict = exposureEntry['CTI']
                if exposureDict[ampName]['IMAGE_MEAN'] < self.config.maxSignalForCti:
                    signal.append(exposureDict[ampName]['IMAGE_MEAN'])
                    data.append(exposureDict[ampName]['OVERSCAN_VALUES'][start:stop+1])
                else:
                    Nskipped += 1
            self.log.info(f"Skipped {Nskipped} exposures brighter than {self.config.maxSignalForCti}.")

            signal = np.array(signal)
            data = np.array(data)

            ind = signal.argsort()
            signal = signal[ind]
            data = data[ind]

            # CTI test.  This looks at the charge that has leaked into
            # the first few columns of the overscan.
            overscan1 = data[:, 0]
            overscan2 = data[:, 1]
            test = (np.array(overscan1) + np.array(overscan2))/(nCols*np.array(signal))
            testResult = np.median(test) > 5.E-6
            self.log.info("Estimate of CTI test is %f for amp %s, %s.", np.median(test), ampName,
                          "full fitting will be performed" if testResult else
                          "only global CTI fitting will be performed")

            self.debugView(ampName, signal, test)

            params = Parameters()
            params.add('ctiexp', value=-6, min=-7, max=-5, vary=True)
            params.add('trapsize', value=5.0 if testResult else 0.0, min=0.0, max=30.,
                       vary=True if testResult else False)
            params.add('scaling', value=0.08, min=0.0, max=1.0,
                       vary=True if testResult else False)
            params.add('emissiontime', value=0.35, min=0.1, max=1.0,
                       vary=True if testResult else False)
            params.add('driftscale', value=calib.driftScale[ampName], min=0., max=0.001, vary=False)
            params.add('decaytime', value=calib.decayTime[ampName], min=0.1, max=4.0, vary=False)

            model = SimulatedModel()
            minner = Minimizer(model.difference, params,
                               fcn_args=(signal, data, self.config.fitError, nCols, amp),
                               fcn_kws={'start': start, 'stop': stop, 'trap_type': 'linear'})
            result = minner.minimize()

            # Only the global CTI term is retained from this fit.
            calib.globalCti[ampName] = 10**result.params['ctiexp'].value
            self.log.info("CTI Global Cti %s: cti: %g decayTime: %g driftScale %g",
                          ampName, calib.globalCti[ampName], calib.decayTime[ampName],
                          calib.driftScale[ampName])

        return calib

    def debugView(self, ampName, signal, test):
        """Debug method for global CTI test value.

        Parameters
        ----------
        ampName : `str`
            Name of the amp for plot title.
        signal : `list` [`float`]
            Image means for the input exposures.
        test : `list` [`float`]
            CTI test value to plot.
        """
        import lsstDebug
        if not lsstDebug.Info(__name__).display:
            return
        if not self.allowDebug:
            return

        import matplotlib.pyplot as plot
        figure = plot.figure(1)
        figure.clear()
        plot.xscale('log', base=10.0)
        plot.yscale('log', base=10.0)
        plot.xlabel('Flat Field Signal [e-?]')
        plot.ylabel('Serial CTI')
        plot.title(ampName)
        plot.plot(signal, test)

        figure.show()
        prompt = "Press Enter or c to continue [chp]..."
        while True:
            ans = input(prompt).lower()
            if ans in ("", " ", "c",):
                break
            elif ans in ("p", ):
                import pdb
                pdb.set_trace()
            elif ans in ('x', ):
                self.allowDebug = False
                break
            elif ans in ("h", ):
                print("[h]elp [c]ontinue [p]db e[x]itDebug")
        plot.close()

    def findTraps(self, inputMeasurements, calib, detector):
        """Solve for serial trap parameters.

        Parameters
        ----------
        inputMeasurements : `list` [`dict`]
            List of overscan measurements from each input exposure.
            Each dictionary is nested within a top level 'CTI' key,
            with measurements organized by amplifier name, containing
            keys:

            ``"FIRST_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"OVERSCAN_COLUMNS"``
                List of overscan column indicies (`list` [`int`]).
            ``"OVERSCAN_VALUES"``
                List of overscan column means (`list` [`float`]).
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Calibration to populate with values.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object containing the geometry information for
            the amplifiers.

        Returns
        -------
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Populated calibration.

        Notes
        -----
        The original CTISIM code uses a data model in which the
        "overscan" consists of the standard serial overscan bbox with
        the values for the last imaging data column prepended to that
        list.  This version of the code keeps the overscan and imaging
        sections separate, and so a -1 offset is needed to ensure that
        the same columns are used for fitting between this code and
        CTISIM.  This offset removes that last imaging data column
        from the count.
        """
        # Range to fit.  These are in "camera" coordinates, and so
        # need to have the count for last image column removed.
        start, stop = self.config.trapColumnRange
        start -= 1
        stop -= 1

        # Loop over amps/inputs, fitting those columns from
        # "non-saturated" inputs.
        for amp in detector.getAmplifiers():
            ampName = amp.getName()

            # Number of serial shifts.
            nCols = amp.getRawDataBBox().getWidth() + amp.getRawSerialPrescanBBox().getWidth()

            # The signal is the mean intensity of each input, and the
            # data are the overscan columns to fit.  The new_signal is
            # the mean in the last image column.  Any serial trap will
            # take charge from this column, and deposit it into the
            # overscan columns.
            signal = []
            data = []
            new_signal = []
            Nskipped = 0
            for exposureEntry in inputMeasurements:
                exposureDict = exposureEntry['CTI']
                if exposureDict[ampName]['IMAGE_MEAN'] < self.config.maxImageMean:
                    signal.append(exposureDict[ampName]['IMAGE_MEAN'])
                    data.append(exposureDict[ampName]['OVERSCAN_VALUES'][start:stop+1])
                    new_signal.append(exposureDict[ampName]['LAST_MEAN'])
                else:
                    Nskipped += 1
            self.log.info(f"Skipped {Nskipped} exposures brighter than {self.config.maxSignalForCti}.")

            signal = np.array(signal)
            data = np.array(data)
            new_signal = np.array(new_signal)

            ind = signal.argsort()
            signal = signal[ind]
            data = data[ind]
            new_signal = new_signal[ind]

            # In the absense of any trap, the model results using the
            # parameters already determined will match the observed
            # overscan results.
            params = Parameters()
            params.add('ctiexp', value=np.log10(calib.globalCti[ampName]),
                       min=-7, max=-5, vary=False)
            params.add('trapsize', value=0.0, min=0.0, max=10., vary=False)
            params.add('scaling', value=0.08, min=0.0, max=1.0, vary=False)
            params.add('emissiontime', value=0.35, min=0.1, max=1.0, vary=False)
            params.add('driftscale', value=calib.driftScale[ampName],
                       min=0.0, max=0.001, vary=False)
            params.add('decaytime', value=calib.decayTime[ampName],
                       min=0.1, max=4.0, vary=False)

            model = SimpleModel.model_results(params, signal, nCols,
                                              start=start, stop=stop)

            # Evaluating trap: the difference between the model and
            # observed data.
            res = np.sum((data-model)[:, :3], axis=1)

            # Create spline model for the trap, using the residual
            # between data and model as a function of the last image
            # column mean (new_signal) scaled by (1 - A_L).
            new_signal = np.asarray((1 - calib.driftScale[ampName])*new_signal, dtype=np.float64)
            x = new_signal
            y = np.maximum(0, res)

            # Pad left with ramp
            y = np.pad(y, (10, 0), 'linear_ramp', end_values=(0, 0))
            x = np.pad(x, (10, 0), 'linear_ramp', end_values=(0, 0))

            # Pad right with constant
            y = np.pad(y, (1, 1), 'constant', constant_values=(0, y[-1]))
            x = np.pad(x, (1, 1), 'constant', constant_values=(-1, 200000.))

            trap = SerialTrap(20000.0, 0.4, 1, 'spline', np.concatenate((x, y)).tolist())
            calib.serialTraps[ampName] = trap

        return calib


class OverscanModel:
    """Base class for handling model/data fit comparisons.

    This handles all of the methods needed for the lmfit Minimizer to
    run.
    """

    @staticmethod
    def model_results(params, signal, num_transfers, start=1, stop=10):
        """Generate a realization of the overscan model, using the specified
        fit parameters and input signal.

        Parameters
        ----------
        params : `lmfit.Parameters`
            Object containing the model parameters.
        signal : `np.ndarray`, (nMeasurements)
            Array of image means.
        num_transfers : `int`
            Number of serial transfers that the charge undergoes.
        start : `int`, optional
            First overscan column to fit. This number includes the
            last imaging column, and needs to be adjusted by one when
            using the overscan bounding box.
        stop : `int`, optional
            Last overscan column to fit. This number includes the
            last imaging column, and needs to be adjusted by one when
            using the overscan bounding box.

        Returns
        -------
        results : `np.ndarray`, (nMeasurements, nCols)
            Model results.
        """
        raise NotImplementedError("Subclasses must implement the model calculation.")

    def loglikelihood(self, params, signal, data, error, *args, **kwargs):
        """Calculate log likelihood of the model.

        Parameters
        ----------
        params : `lmfit.Parameters`
            Object containing the model parameters.
        signal : `np.ndarray`, (nMeasurements)
            Array of image means.
        data : `np.ndarray`, (nMeasurements, nCols)
            Array of overscan column means from each measurement.
        error : `float`
            Fixed error value.
        *args :
            Additional position arguments.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        logL : `float`
            The log-likelihood of the observed data given the model
            parameters.
        """
        model_results = self.model_results(params, signal, *args, **kwargs)

        inv_sigma2 = 1.0/(error**2.0)
        diff = model_results - data

        return -0.5*(np.sum(inv_sigma2*(diff)**2.))

    def negative_loglikelihood(self, params, signal, data, error, *args, **kwargs):
        """Calculate negative log likelihood of the model.

        Parameters
        ----------
        params : `lmfit.Parameters`
            Object containing the model parameters.
        signal : `np.ndarray`, (nMeasurements)
            Array of image means.
        data : `np.ndarray`, (nMeasurements, nCols)
            Array of overscan column means from each measurement.
        error : `float`
            Fixed error value.
        *args :
            Additional position arguments.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        negativelogL : `float`
            The negative log-likelihood of the observed data given the
            model parameters.
        """
        ll = self.loglikelihood(params, signal, data, error, *args, **kwargs)

        return -ll

    def rms_error(self, params, signal, data, error, *args, **kwargs):
        """Calculate RMS error between model and data.

        Parameters
        ----------
        params : `lmfit.Parameters`
            Object containing the model parameters.
        signal : `np.ndarray`, (nMeasurements)
            Array of image means.
        data : `np.ndarray`, (nMeasurements, nCols)
            Array of overscan column means from each measurement.
        error : `float`
            Fixed error value.
        *args :
            Additional position arguments.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        rms : `float`
            The rms error between the model and input data.
        """
        model_results = self.model_results(params, signal, *args, **kwargs)

        diff = model_results - data
        rms = np.sqrt(np.mean(np.square(diff)))

        return rms

    def difference(self, params, signal, data, error, *args, **kwargs):
        """Calculate the flattened difference array between model and data.

        Parameters
        ----------
        params : `lmfit.Parameters`
            Object containing the model parameters.
        signal : `np.ndarray`, (nMeasurements)
            Array of image means.
        data : `np.ndarray`, (nMeasurements, nCols)
            Array of overscan column means from each measurement.
        error : `float`
            Fixed error value.
        *args :
            Additional position arguments.
        **kwargs :
            Additional keyword arguments.

        Returns
        -------
        difference : `np.ndarray`, (nMeasurements*nCols)
            The rms error between the model and input data.
        """
        model_results = self.model_results(params, signal, *args, **kwargs)
        diff = (model_results-data).flatten()

        return diff


class SimpleModel(OverscanModel):
    """Simple analytic overscan model."""

    @staticmethod
    def model_results(params, signal, num_transfers, start=1, stop=10):
        """Generate a realization of the overscan model, using the specified
        fit parameters and input signal.

        Parameters
        ----------
        params : `lmfit.Parameters`
            Object containing the model parameters.
        signal : `np.ndarray`, (nMeasurements)
            Array of image means.
        num_transfers : `int`
            Number of serial transfers that the charge undergoes.
        start : `int`, optional
            First overscan column to fit. This number includes the
            last imaging column, and needs to be adjusted by one when
            using the overscan bounding box.
        stop : `int`, optional
            Last overscan column to fit. This number includes the
            last imaging column, and needs to be adjusted by one when
            using the overscan bounding box.

        Returns
        -------
        res : `np.ndarray`, (nMeasurements, nCols)
            Model results.
        """
        v = params.valuesdict()
        v['cti'] = 10**v['ctiexp']

        # Adjust column numbering to match DM overscan bbox.
        start += 1
        stop += 1

        x = np.arange(start, stop+1)
        res = np.zeros((signal.shape[0], x.shape[0]))

        for i, s in enumerate(signal):
            # This is largely equivalent to equation 2.  The minimum
            # indicates that a trap cannot emit more charge than is
            # available, nor can it emit more charge than it can hold.
            # This scales the exponential release of charge from the
            # trap.  The next term defines the contribution from the
            # global CTI at each pixel transfer, and the final term
            # includes the contribution from local CTI effects.
            res[i, :] = (np.minimum(v['trapsize'], s*v['scaling'])
                         * (np.exp(1/v['emissiontime']) - 1.0)
                         * np.exp(-x/v['emissiontime'])
                         + s*num_transfers*v['cti']**x
                         + v['driftscale']*s*np.exp(-x/float(v['decaytime'])))

        return res


class SimulatedModel(OverscanModel):
    """Simulated overscan model."""

    @staticmethod
    def model_results(params, signal, num_transfers, amp, start=1, stop=10, trap_type=None):
        """Generate a realization of the overscan model, using the specified
        fit parameters and input signal.

        Parameters
        ----------
        params : `lmfit.Parameters`
            Object containing the model parameters.
        signal : `np.ndarray`, (nMeasurements)
            Array of image means.
        num_transfers : `int`
            Number of serial transfers that the charge undergoes.
        amp : `lsst.afw.cameraGeom.Amplifier`
            Amplifier to use for geometry information.
        start : `int`, optional
            First overscan column to fit. This number includes the
            last imaging column, and needs to be adjusted by one when
            using the overscan bounding box.
        stop : `int`, optional
            Last overscan column to fit. This number includes the
            last imaging column, and needs to be adjusted by one when
            using the overscan bounding box.
        trap_type : `str`, optional
            Type of trap model to use.

        Returns
        -------
        results : `np.ndarray`, (nMeasurements, nCols)
            Model results.
        """
        v = params.valuesdict()

        # Adjust column numbering to match DM overscan bbox.
        start += 1
        stop += 1

        # Electronics effect optimization
        output_amplifier = FloatingOutputAmplifier(1.0, v['driftscale'], v['decaytime'])

        # CTI optimization
        v['cti'] = 10**v['ctiexp']

        # Trap type for optimization
        if trap_type is None:
            trap = None
        elif trap_type == 'linear':
            trap = SerialTrap(v['trapsize'], v['emissiontime'], 1, 'linear',
                              [v['scaling']])
        elif trap_type == 'logistic':
            trap = SerialTrap(v['trapsize'], v['emissiontime'], 1, 'logistic',
                              [v['f0'], v['k']])
        else:
            raise ValueError('Trap type must be linear or logistic or None')

        # Simulate ramp readout
        imarr = np.zeros((signal.shape[0], amp.getRawDataBBox().getWidth()))
        ramp = SegmentSimulator(imarr, amp.getRawSerialPrescanBBox().getWidth(), output_amplifier,
                                cti=v['cti'], traps=trap)
        ramp.ramp_exp(signal)
        model_results = ramp.readout(serial_overscan_width=amp.getRawSerialOverscanBBox().getWidth(),
                                     parallel_overscan_width=0)

        ncols = amp.getRawSerialPrescanBBox().getWidth() + amp.getRawDataBBox().getWidth()

        return model_results[:, ncols+start-1:ncols+stop]


class SegmentSimulator:
    """Controls the creation of simulated segment images.

    Parameters
    ----------
    imarr : `np.ndarray` (nx, ny)
        Image data array.
    prescan_width : `int`
        Number of serial prescan columns.
    output_amplifier : `lsst.cp.pipe.FloatingOutputAmplifier`
        An object holding the gain, read noise, and global_offset.
    cti : `float`
        Global CTI value.
    traps : `list` [`lsst.ip.isr.SerialTrap`]
        Serial traps to simulate.
    """

    def __init__(self, imarr, prescan_width, output_amplifier, cti=0.0, traps=None):
        # Image array geometry
        self.prescan_width = prescan_width
        self.ny, self.nx = imarr.shape

        self.segarr = np.zeros((self.ny, self.nx+prescan_width))
        self.segarr[:, prescan_width:] = imarr

        # Serial readout information
        self.output_amplifier = output_amplifier
        if isinstance(cti, np.ndarray):
            raise ValueError("cti must be single value, not an array.")
        self.cti = cti

        self.serial_traps = None
        self.do_trapping = False
        if traps is not None:
            if not isinstance(traps, list):
                traps = [traps]
            for trap in traps:
                self.add_trap(trap)

    def add_trap(self, serial_trap):
        """Add a trap to the serial register.

        Parameters
        ----------
        serial_trap : `lsst.ip.isr.SerialTrap`
            The trap to add.
        """
        try:
            self.serial_traps.append(serial_trap)
        except AttributeError:
            self.serial_traps = [serial_trap]
            self.do_trapping = True

    def ramp_exp(self, signal_list):
        """Simulate an image with varying flux illumination per row.

        This method simulates a segment image where the signal level
        increases along the horizontal direction, according to the
        provided list of signal levels.

        Parameters
        ----------
        signal_list : `list` [`float`]
            List of signal levels.

        Raises
        ------
        ValueError
            Raised if the length of the signal list does not equal the
            number of rows.
        """
        if len(signal_list) != self.ny:
            raise ValueError("Signal list does not match row count.")

        ramp = np.tile(signal_list, (self.nx, 1)).T
        self.segarr[:, self.prescan_width:] += ramp

    def readout(self, serial_overscan_width=10, parallel_overscan_width=0):
        """Simulate serial readout of the segment image.

        This method performs the serial readout of a segment image
        given the appropriate SerialRegister object and the properties
        of the ReadoutAmplifier.  Additional arguments can be provided
        to account for the number of desired overscan transfers. The
        result is a simulated final segment image, in ADU.

        Parameters
        ----------
        serial_overscan_width : `int`, optional
            Number of serial overscan columns.
        parallel_overscan_width : `int`, optional
            Number of parallel overscan rows.

        Returns
        -------
        result : `np.ndarray` (nx, ny)
            Simulated image, including serial prescan, serial
            overscan, and parallel overscan regions.
        """
        # Create output array
        iy = int(self.ny + parallel_overscan_width)
        ix = int(self.nx + self.prescan_width + serial_overscan_width)
        image = np.random.normal(loc=self.output_amplifier.global_offset,
                                 scale=self.output_amplifier.noise,
                                 size=(iy, ix))
        free_charge = copy.deepcopy(self.segarr)

        # Set flow control parameters
        do_trapping = self.do_trapping
        cti = self.cti

        offset = np.zeros(self.ny)
        cte = 1 - cti
        if do_trapping:
            for trap in self.serial_traps:
                trap.initialize(self.ny, self.nx, self.prescan_width)

        for i in range(ix):
            # Trap capture
            if do_trapping:
                for trap in self.serial_traps:
                    captured_charge = trap.trap_charge(free_charge)
                    free_charge -= captured_charge

            # Pixel-to-pixel proportional loss
            transferred_charge = free_charge*cte
            deferred_charge = free_charge*cti

            # Pixel transfer and readout
            offset = self.output_amplifier.local_offset(offset,
                                                        transferred_charge[:, 0])
            image[:iy-parallel_overscan_width, i] += transferred_charge[:, 0] + offset

            free_charge = np.pad(transferred_charge, ((0, 0), (0, 1)),
                                 mode='constant')[:, 1:] + deferred_charge

            # Trap emission
            if do_trapping:
                for trap in self.serial_traps:
                    released_charge = trap.release_charge()
                    free_charge += released_charge

        return image/float(self.output_amplifier.gain)


class FloatingOutputAmplifier:
    """Object representing the readout amplifier of a single channel.

    Parameters
    ----------
    gain : `float`
        Amplifier gain.
    scale : `float`
        Drift scale for the amplifier.
    decay_time : `float`
        Decay time for the bias drift.
    noise : `float`, optional
        Amplifier read noise.
    offset : `float`, optional
        Global CTI offset.
    """

    def __init__(self, gain, scale, decay_time, noise=0.0, offset=0.0):

        self.gain = gain
        self.noise = noise
        self.global_offset = offset

        self.update_parameters(scale, decay_time)

    def local_offset(self, old, signal):
        """Calculate local offset hysteresis.

        Parameters
        ----------
        old : `np.ndarray`, (,)
            Previous iteration.
        signal : `np.ndarray`, (,)
            Current column measurements.

        Returns
        -------
        offset : `np.ndarray`
            Local offset.
        """
        new = self.scale*signal

        return np.maximum(new, old*np.exp(-1/self.decay_time))

    def update_parameters(self, scale, decay_time):
        """Update parameter values, if within acceptable values.

        Parameters
        ----------
        scale : `float`
            Drift scale for the amplifier.
        decay_time : `float`
            Decay time for the bias drift.

        Raises
        ------
        ValueError
            Raised if the input parameters are out of range.
        """
        if scale < 0.0:
            raise ValueError("Scale must be greater than or equal to 0.")
        if np.isnan(scale):
            raise ValueError("Scale must be real-valued number, not NaN.")
        self.scale = scale
        if decay_time <= 0.0:
            raise ValueError("Decay time must be greater than 0.")
        if np.isnan(decay_time):
            raise ValueError("Decay time must be real-valued number, not NaN.")
        self.decay_time = decay_time
