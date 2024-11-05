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
__all__ = ('CpCtiSolveConnections',
           'CpCtiSolveConfig',
           'CpCtiSolveTask',
           )

import numpy as np

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from lsst.ip.isr.deferredCharge import (DeferredChargeCalib,
                                        SimpleModel,
                                        SimulatedModel,
                                        SerialTrap)
from lmfit import Minimizer, Parameters


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
        isCalibration=True,
    )

    outputCalib = cT.Output(
        name="cpCtiCalib",
        doc="Output CTI calibration.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )


class CpCtiSolveConfig(pipeBase.PipelineTaskConfig,
                       pipelineConnections=CpCtiSolveConnections):
    """Configuration for the CTI combination.
    """
    maxImageMean = pexConfig.Field(
        dtype=float,
        default=150000.0,
        doc="Upper limit on acceptable image flux mean (electron).",
    )
    localOffsetColumnRange = pexConfig.ListField(
        dtype=int,
        default=[3, 13],
        doc="First and last overscan column to use for local offset effect.",
    )

    useGains = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Use gains in calculation.",
        deprecated="This field is no longer used. Will be removed after v28.",
    )
    maxSignalForCti = pexConfig.Field(
        dtype=float,
        default=10000.0,
        doc="Upper flux limit to use for CTI fit (electron).",
    )
    globalCtiColumnRange = pexConfig.ListField(
        dtype=int,
        default=[1, 2],
        doc="First and last serialoverscan column to use for "
            "global CTI fit.",
    )
    globalCtiRowRange = pexConfig.ListField(
        dtype=int,
        default=[1, 2],
        doc="First and last parallel overscan row to use for "
            "global CTI fit.",
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


class CpCtiSolveTask(pipeBase.PipelineTask):
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

        dimensions = [dict(exp.dataId.required) for exp in inputRefs.inputMeasurements]
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

            ``"FIRST_COLUMN_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_COLUMN_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"SERIAL_OVERSCAN_COLUMNS"``
                List of overscan column indicies (`list` [`int`]).
            ``"SERIAL_OVERSCAN_VALUES"``
                List of overscan column means (`list` [`float`]).
            ``"PARALLEL_OVERSCAN_ROWS"``
                List of overscan row indicies (`list` [`int`]).
            ``"PARALLEL_OVERSCAN_VALUES"``
                List of overscan row means (`list` [`float`).
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

            ``"FIRST_COLUMN_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_COLUMN_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"SERIAL_OVERSCAN_COLUMNS"``
                List of overscan column indicies (`list` [`int`]).
            ``"SERIAL_OVERSCAN_VALUES"``
                List of overscan column means (`list` [`float`]).
            ``"PARALLEL_OVERSCAN_ROWS"``
                List of overscan row indicies (`list` [`int`]).
            ``"PARALLEL_OVERSCAN_VALUES"``
                List of overscan row means (`list` [`float`).
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Calibration to populate with values.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object containing the geometry information for
            the amplifiers.

        Returns
        -------
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Populated calibration.

        Raises
        ------
        RuntimeError
            Raised if no data remains after flux filtering.

        Notes
        -----
        The original CTISIM code (https://github.com/Snyder005/ctisim)
        uses a data model in which the "overscan" consists of the
        standard serial overscan bbox with the values for the last
        imaging data column prepended to that list.  This version of
        the code keeps the overscan and imaging sections separate, and
        so a -1 offset is needed to ensure that the same columns are
        used for fitting between this code and CTISIM.  This offset
        removes that last imaging data column from the count.
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
                    data.append(exposureDict[ampName]['SERIAL_OVERSCAN_VALUES'][start:stop+1])
                else:
                    Nskipped += 1
            self.log.info(f"Skipped {Nskipped} exposures brighter than {self.config.maxImageMean}.")
            if len(signal) == 0 or len(data) == 0:
                raise RuntimeError("All exposures brighter than config.maxImageMean and excluded.")

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

            ``"FIRST_COLUMN_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_COLUMN_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"SERIAL_OVERSCAN_COLUMNS"``
                List of overscan column indicies (`list` [`int`]).
            ``"SERIAL_OVERSCAN_VALUES"``
                List of overscan column means (`list` [`float`]).
            ``"PARALLEL_OVERSCAN_ROWS"``
                List of overscan row indicies (`list` [`int`]).
            ``"PARALLEL_OVERSCAN_VALUES"``
                List of overscan row means (`list` [`float`).
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Calibration to populate with values.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector object containing the geometry information for
            the amplifiers.

        Returns
        -------
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Populated calibration.

        Raises
        ------
        RuntimeError
            Raised if no data remains after flux filtering.

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
        # Loop over amps/inputs, fitting those columns from
        # "non-saturated" inputs.
        for amp in detector.getAmplifiers():
            ampName = amp.getName()

            # Number of serial shifts.
            nCols = amp.getRawDataBBox().getWidth() + amp.getRawSerialPrescanBBox().getWidth()

            # Do serial CTI calculation
            signals, data, start, stop, serialEperEstimate = self.calcEper("SERIAL", inputMeasurements, amp)
            serialEperResult = np.median(serialEperEstimate) > 5.E-6
            self.log.info(
                "Estimate of CTI test is %f for amp %s, %s.", np.median(serialEperEstimate), ampName,
                "full fitting will be performed" if serialEperResult else
                "only global CTI fitting will be performed"
            )

            self.debugView(ampName, signals, serialEperEstimate)

            params = Parameters()
            params.add('ctiexp', value=-6, min=-7, max=-5, vary=True)
            params.add('trapsize', value=5.0 if serialEperResult else 0.0, min=0.0, max=30.,
                       vary=True if serialEperResult else False)
            params.add('scaling', value=0.08, min=0.0, max=1.0,
                       vary=True if serialEperResult else False)
            params.add('emissiontime', value=0.35, min=0.1, max=1.0,
                       vary=True if serialEperResult else False)
            params.add('driftscale', value=calib.driftScale[ampName], min=0., max=0.001, vary=False)
            params.add('decaytime', value=calib.decayTime[ampName], min=0.1, max=4.0, vary=False)

            model = SimulatedModel()
            minner = Minimizer(model.difference, params,
                               fcn_args=(signals, data, self.config.fitError, nCols, amp),
                               fcn_kws={'start': start, 'stop': stop, 'trap_type': 'linear'})
            result = minner.minimize()

            # Only the global CTI term is retained from this fit.
            calib.globalCti[ampName] = 10**result.params['ctiexp'].value
            self.log.info("CTI Global Cti %s: cti: %g decayTime: %g driftScale %g",
                          ampName, calib.globalCti[ampName], calib.decayTime[ampName],
                          calib.driftScale[ampName])

            # Do parallel EPER calculation
            signals, data, start, stop, parallelEperEstimate = self.calcEper(
                "PARALLEL",
                inputMeasurements,
                amp,
            )

            calib.signals[ampName] = signals
            calib.serialEper[ampName] = serialEperEstimate
            calib.parallelEper[ampName] = parallelEperEstimate

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

            ``"FIRST_COLUMN_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_COLUMN_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"SERIAL_OVERSCAN_COLUMNS"``
                List of overscan column indicies (`list` [`int`]).
            ``"SERIAL_OVERSCAN_VALUES"``
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

        Raises
        ------
        RuntimeError
            Raised if no data remains after flux filtering.

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
                    data.append(exposureDict[ampName]['SERIAL_OVERSCAN_VALUES'][start:stop+1])
                    new_signal.append(exposureDict[ampName]['LAST_COLUMN_MEAN'])
                else:
                    Nskipped += 1
            self.log.info(f"Skipped {Nskipped} exposures brighter than {self.config.maxImageMean}.")
            if len(signal) == 0 or len(data) == 0:
                raise RuntimeError("All exposures brighter than config.maxImageMean and excluded.")

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
            # Note that this ``spline`` model is actually a piecewise
            # linear interpolation and not a true spline.
            new_signal = np.asarray((1 - calib.driftScale[ampName])*new_signal, dtype=np.float64)
            x = new_signal
            y = np.maximum(0, res)

            # Pad left with ramp
            y = np.pad(y, (10, 0), 'linear_ramp', end_values=(0, 0))
            x = np.pad(x, (10, 0), 'linear_ramp', end_values=(0, 0))

            trap = SerialTrap(20000.0, 0.4, 1, 'spline', np.concatenate((x, y)).tolist())
            calib.serialTraps[ampName] = trap

        return calib

    def calcEper(self, mode, inputMeasurements, amp):
        """Solve for serial global CTI using the extended pixel edge
        response (EPER) method.

        Parameters
        ----------
        mode : `str`
            The orientation of the calculation to perform. Can be
            either `SERIAL` or `PARALLEL`.
        inputMeasurements : `list` [`dict`]
            List of overscan measurements from each input exposure.
            Each dictionary is nested within a top level 'CTI' key,
            with measurements organized by amplifier name, containing
            keys:

            ``"FIRST_COLUMN_MEAN"``
                Mean value of first image column (`float`).
            ``"LAST_COLUMN_MEAN"``
                Mean value of last image column (`float`).
            ``"IMAGE_MEAN"``
                Mean value of the entire image region (`float`).
            ``"SERIAL_OVERSCAN_COLUMNS"``
                List of serial overscan column
                indicies (`list` [`int`]).
            ``"SERIAL_OVERSCAN_VALUES"``
                List of serial overscan column
                means (`list` [`float`]).
            ``"PARALLEL_OVERSCAN_ROWS"``
                List of parallel overscan row
                indicies (`list` [`int`]).
            ``"PARALLEL_OVERSCAN_VALUES"``
                List of parallel overscan row
                means (`list` [`float`]).
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Calibration to populate with values.
        amp : `lsst.afw.cameraGeom.Amplifier`
            Amplifier object containing the geometry information for
            the amplifier.

        Returns
        -------
        calib : `lsst.ip.isr.DeferredChargeCalib`
            Populated calibration.

        Raises
        ------
        RuntimeError : Raised if no data remains after flux filtering or if
            the mode string is not one of "SERIAL" or "PARALLEL".
        """
        ampName = amp.getName()

        # Range to fit.  These are in "camera" coordinates, and so
        # need to have the count for last image column removed.
        if mode == "SERIAL":
            start, stop = self.config.globalCtiColumnRange
            start -= 1
            stop -= 1

            # Number of serial shifts = nCols
            nShifts = amp.getRawDataBBox().getWidth() + amp.getRawSerialPrescanBBox().getWidth()
        elif mode == "PARALLEL":
            start, stop = self.config.globalCtiRowRange
            start -= 1
            stop -= 1

            # Number of parallel shifts = nRows
            nShifts = amp.getRawDataBBox().getHeight()
        else:
            raise RuntimeError(f"{mode} is not a known orientation for the EPER calculation.")

        # The signal is the mean intensity of each input, and the
        # data are the overscan columns to fit.  For detectors
        # with non-zero CTI, the charge from the imaging region
        # leaks into the overscan region.
        signal = []
        data = []
        Nskipped = 0
        for idx, exposureEntry in enumerate(inputMeasurements):
            exposureDict = exposureEntry['CTI']
            if exposureDict[ampName]['IMAGE_MEAN'] < self.config.maxSignalForCti:
                signal.append(exposureDict[ampName]['IMAGE_MEAN'])
                data.append(exposureDict[ampName][f'{mode}_OVERSCAN_VALUES'][start:stop+1])
            else:
                Nskipped += 1
        self.log.info(f"Skipped {Nskipped} exposures brighter than {self.config.maxSignalForCti}.")
        if len(signal) == 0 or len(data) == 0:
            raise RuntimeError("All exposures brighter than config.maxSignalForCti and excluded.")

        signal = np.array(signal)
        data = np.array(data)

        ind = signal.argsort()
        signal = signal[ind]
        data = data[ind]

        # This looks at the charge that has leaked into
        # the first few columns of the overscan.
        overscan1 = data[:, 0]
        overscan2 = data[:, 1]
        ctiEstimate = (np.array(overscan1) + np.array(overscan2))/(nShifts*np.array(signal))

        return signal, data, start, stop, ctiEstimate
