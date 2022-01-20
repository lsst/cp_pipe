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

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig

from lsst.ip.isr import DeferredChargeCalib, SerialTrap
from lmfit import Minimizer, Parameters

from ._lookupStaticCalibration import lookupStaticCalibration


class CpCtiSolveConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("instrument", "detector")):
    inputMeasurements = cT.Input(
        name="cpCtiMeas",
        doc="Input measurements to fit.",
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
        doc="Output measurements.",
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
        doc="Upper limit on acceptable image means.",
    )
    localOffsetColumnRange = pexConfig.ListField(
        dtype=int,
        default=[3, 13],
        doc="First and last overscan column to use for local offset effect.",
    )

    maxSignalForCti = pexConfig.Field(
        dtype=float,
        default=10000.0,
        doc="Upper limit to use for CTI fit.",
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
        dtype=float,
        default=7.0/np.sqrt(2000),
        doc="Error to use during parameter fitting.",
    )


class CpCtiSolveTask(pipeBase.PipelineTask,
                     pipeBase.CmdLineTask):
    """Combine CTI measurements to a final calibration.
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
        """
        """
        detectorSet = sorted(set([d['detector'] for d in inputDims]))
        if len(detectorSet) != 1:
            raise RuntimeError("Inputs for too many detectors passed.")
        detectorId = detectorSet[0]
        detector = camera[detectorId]

        # Initialize with detector.
        calib = DeferredChargeCalib(camera=camera, detector=detector)

        localCalib = self.localOffsets(inputMeasurements, calib, detector)

        globalCalib = self.globalCti(inputMeasurements, localCalib, detector)

        finalCalib = self.findTraps(inputMeasurements, globalCalib, detector)

        return pipeBase.Struct(
            outputCalib=finalCalib,
        )

    def localOffsets(self, inputMeasurements, calib, detector):
        # Range to fit.
        start, stop = self.config.localOffsetColumnRange

        # Loop over amps/inputs, fitting those columns from
        # "non-saturated" inputs.
        for amp in detector.getAmplifiers():
            ampName = amp.getName()

            # Number of serial shifts.
            nCols = amp.getRawDataBBox().getWidth() + amp.getRawSerialPrescanBBox().getWidth()

            signal = []
            data = []
            for exposureEntry in inputMeasurements:
                exposureDict = exposureEntry['CTI']
                if exposureDict[ampName]['IMAGE_MEAN'] < self.config.maxImageMean:
                    data.append(np.flip(exposureDict[ampName]['OVERSCAN_VALUES'])[start:stop+1])
                    signal.append(exposureDict[ampName]['IMAGE_MEAN'])

            signal = np.array(signal)
            data = np.array(data)

            ind = signal.argsort()
            signal = signal[ind]
            data = data[ind]

            params = Parameters()
            params.add('ctiexp', value=-6, min=-7, max=-5, vary=False)
            params.add('trapsize', value=0.0, min=0., max=10., vary=False)
            params.add('scaling', value=0.08, min=0, max=1.0, vary=False)
            params.add('emissiontime', value=0.4, min=0.1, max=1.0, vary=False)
            params.add('driftscale', value=0.00022, min=0., max=0.001)
            params.add('decaytime', value=2.4, min=0.1, max=4.0)

            model = SimpleModel()

            minner = Minimizer(model.difference, params,
                               fcn_args=(signal, data, self.config.fitError, nCols),
                               fcn_kws={'start': start, 'stop': stop})
            result = minner.minimize()

            # Save results
            if not result.success:
                self.log("Electronics fitting failure for amplifier %s.", ampName)

            calib.globalCti[ampName] = 10**result.params['ctiexp']
            calib.decayTime[ampName] = result.params['driftscale'].value if result.success else 2.4
            calib.driftScale[ampName] = result.params['decaytime'].value if result.success else 0.0
            self.log.info("CTI Local Fit %s: cti: %g decayTime: %g driftScale %g",
                          ampName, calib.globalCti[ampName], calib.decayTime[ampName],
                          calib.driftScale[ampName])
        return calib

    def globalCti(self, inputMeasurements, calib, detector):
        """ XXX """
        # Range to fit.
        start, stop = self.config.globalCtiColumnRange

        # Loop over amps/inputs, fitting those columns from
        # "non-saturated" inputs.
        for amp in detector.getAmplifiers():
            ampName = amp.getName()

            # Number of serial shifts.
            nCols = amp.getRawDataBBox().getWidth() + amp.getRawSerialPrescanBBox().getWidth()

            signal = []
            data = []
            for exposureEntry in inputMeasurements:
                exposureDict = exposureEntry['CTI']
                if exposureDict[ampName]['IMAGE_MEAN'] < self.config.maxSignalForCti:
                    signal.append(exposureDict[ampName]['IMAGE_MEAN'])
                    data.append(np.flip(exposureDict[ampName]['OVERSCAN_VALUES'])[start:stop+1])

            signal = np.array(signal)
            data = np.array(data)

            ind = signal.argsort()
            signal = signal[ind]
            data = data[ind]

            # CTI test
            overscan1 = data[:, 0]
            overscan2 = data[:, 1]
            test = (np.array(overscan1) + np.array(overscan2))/(nCols*np.array(signal))
            testResult = np.median(test) > 5.E-6
            self.debugView(ampName, signal, test)

            params = Parameters()
            params.add('ctiexp', value=-6, min=-7, max=np.log10(1E-5), vary=True)
            params.add('trapsize', value=5.0, min=0., max=30., vary=True if testResult else False)
            params.add('scaling', value=0.08, min=0, max=1.0, vary=True if testResult else False)
            params.add('emissiontime', value=0.35, min=0.1, max=1.0, vary=True if testResult else False)
            params.add('driftscale', value=calib.driftScale[ampName], min=0., max=0.001, vary=False)
            params.add('decaytime', value=calib.decayTime[ampName], min=0.1, max=4.0, vary=False)

            model = SimulatedModel()
            minner = Minimizer(model.difference, params,
                               fcn_args=(signal, data, self.config.fitError, nCols, amp),
                               fcn_kws={'start': start, 'stop': stop, 'trap_type': 'linear'})
            result = minner.minimize()

            calib.globalCti[ampName] = 10**result.params['ctiexp'].value
            self.log.info("CTI Global Cti %s: cti: %g decayTime: %g driftScale %g",
                          ampName, calib.globalCti[ampName], calib.decayTime[ampName],
                          calib.driftScale[ampName])

        return calib

    def debugView(self, ampName, signal, test):
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
        """ XXX """
        start, stop = self.config.trapColumnRange

        # Loop over amps/inputs, fitting those columns from
        # "non-saturated" inputs.
        for amp in detector.getAmplifiers():
            ampName = amp.getName()

            # Number of serial shifts.
            nCols = amp.getRawDataBBox().getWidth() + amp.getRawSerialPrescanBBox().getWidth()

            signal = []
            data = []
            new_signal = []
            for exposureEntry in inputMeasurements:
                exposureDict = exposureEntry['CTI']
                if exposureDict[ampName]['IMAGE_MEAN'] < self.config.maxImageMean:
                    signal.append(exposureDict[ampName]['IMAGE_MEAN'])
                    data.append(np.flip(exposureDict[ampName]['OVERSCAN_VALUES'])[start:stop+1])
                    new_signal.append(np.flip(exposureDict[ampName]['OVERSCAN_VALUES'])[0])

            signal = np.array(signal)
            data = np.array(data)
            new_signal = np.array(new_signal)

            ind = signal.argsort()
            signal = signal[ind]
            data = data[ind]
            new_signal = new_signal[ind]

            # Second model: model with electronics
            params = Parameters()
            params.add('ctiexp', value=np.log10(calib.globalCti[ampName]),
                       min=-7, max=-5, vary=False)
            params.add('trapsize', value=0.0, min=0., max=10., vary=False)
            params.add('scaling', value=0.08, min=0, max=1.0, vary=False)
            params.add('emissiontime', value=0.35, min=0.1, max=1.0, vary=False)
            params.add('driftscale', value=calib.driftScale[ampName],
                       min=0., max=0.001, vary=False)
            params.add('decaytime', value=calib.decayTime[ampName],
                       min=0.1, max=4.0, vary=False)

            model = SimpleModel.model_results(params, signal, nCols,
                                              start=start, stop=stop)

            # Evaluating trap
            res = np.sum((data-model)[:, :3], axis=1)

            rescale = calib.driftScale[ampName]*signal
            new_signal = np.asarray(signal - rescale, dtype=np.float64)
            x = signal
            y = np.maximum(0, res)

            # Pad left with ramp
            y = np.pad(y, (10, 0), 'linear_ramp', end_values=(0, 0))
            x = np.pad(x, (10, 0), 'linear_ramp', end_values=(0, 0))

            # Pad right with constant
            y = np.pad(y, (1, 1), 'constant', constant_values=(0, y[-1]))
            x = np.pad(x, (1, 1), 'constant', constant_values=(-1, 200000.))
            import pdb; pdb.set_trace()
            trap = SerialTrap(20000.0, 0.4, 1, 'spline', np.concatenate((x, y)).tolist())
            calib.serialTraps[ampName] = trap

        return calib


class OverscanModel:
    """Base object handling model/data fit comparisons."""

    def loglikelihood(self, params, signal, data, error,
                      *args, **kwargs):
        """Calculate log likelihood of the model."""

        model_results = self.model_results(params, signal,
                                           *args, **kwargs)

        inv_sigma2 = 1./(error**2.)
        diff = model_results-data

        return -0.5*(np.sum(inv_sigma2*(diff)**2.))

    def negative_loglikelihood(self, params, signal, data, error,
                               *args, **kwargs):
        """Calculate negative log likelihood of the model."""

        ll = self.loglikelihood(params, signal, data, error, *args, **kwargs)

        return -ll

    def rms_error(self, params, signal, data, error, *args, **kwargs):
        """Calculate RMS error between model and data."""

        model_results = self.model_results(params, signal, *args, **kwargs)

        diff = model_results - data

        rms = np.sqrt(np.mean(np.square(diff)))

        return rms

    def difference(self, params, signal, data, error, *args, **kwargs):
        """Calculate the flattened difference array between model and data."""

        model_results = self.model_results(params, signal, *args, **kwargs)

        diff = (model_results-data).flatten()

        return diff


class SimpleModel(OverscanModel):
    """Simple analytic overscan model."""

    @staticmethod
    def model_results(params, signal, num_transfers, start=1, stop=10):
        v = params.valuesdict()
        try:
            v['cti'] = 10**v['ctiexp']
        except KeyError:
            pass

        x = np.arange(start, stop+1)
        res = np.zeros((signal.shape[0], x.shape[0]))

        for i, s in enumerate(signal):
            res[i, :] = (np.minimum(v['trapsize'], s*v['scaling'])*(np.exp(1/v['emissiontime']) - 1.)
                         * np.exp(-x/v['emissiontime'])
                         + s*num_transfers*v['cti']**x
                         + v['driftscale']*s*np.exp(-x/float(v['decaytime'])))

        return res


class SimulatedModel(OverscanModel):
    """Simulated overscan model."""

    @staticmethod
    def model_results(params, signal, num_transfers, amp, **kwargs):
        v = params.valuesdict()

        start = kwargs.pop('start', 1)
        stop = kwargs.pop('stop', 10)
        trap_type = kwargs.pop('trap_type', None)

        # Electronics effect optimization
        if 'beta' in v.keys():
            output_amplifier = FloatingOutputAmplifier2(1.0,
                                                        v['driftscale'],
                                                        v['decaytime'],
                                                        v['beta'])

        elif 'driftscale' in v.keys():
            output_amplifier = FloatingOutputAmplifier(1.0,
                                                       v['driftscale'],
                                                       v['decaytime'])
        else:
            output_amplifier = BaseOutputAmplifier(1.0)

        # CTI optimization
        try:
            v['cti'] = 10**v['ctiexp']
        except KeyError:
            pass

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

        # Optional fixed traps
        try:
            fixed_traps = kwargs['fixed_traps']
            if isinstance(trap, list):
                trap.append(fixed_traps)
            else:
                trap = [fixed_traps, trap]
        except KeyError:
            trap = trap

        # Simulate ramp readout
        imarr = np.zeros((signal.shape[0], amp.getRawDataBBox().getWidth()))
        ramp = SegmentSimulator(imarr, amp.getRawSerialPrescanBBox().getWidth(), output_amplifier,
                                cti=v['cti'], traps=trap)
        ramp.ramp_exp(signal)
        model_results = ramp.readout(serial_overscan_width=amp.getRawSerialOverscanBBox().getWidth(),
                                     parallel_overscan_width=0, **kwargs)

        ncols = amp.getRawSerialPrescanBBox().getWidth() + amp.getRawDataBBox().getWidth()

        return model_results[:, ncols+start-1:ncols+stop]


class SegmentSimulator:
    """Controls the creation of simulated segment images.

    Attributes:
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        num_serial_prescan (int): Number of serial prescan pixels.
        image (numpy.array): NumPy array containg the image pixels.
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

        Args: serial_trap (SerialTrap): Serial trap to include in
            serial register.

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

        Args:
            signal_list ('list' of 'float'): List of signal levels.

        Raises: ValueError: If number of signal levels does not equal
            the number of rows.

        """
        if len(signal_list) != self.ny:
            raise ValueError

        ramp = np.tile(signal_list, (self.nx, 1)).T
        self.segarr[:, self.prescan_width:] += ramp

    def readout(self, serial_overscan_width=10, parallel_overscan_width=0, **kwargs):
        """Simulate serial readout of the segment image.

        This method performs the serial readout of a segment image
        given the appropriate SerialRegister object and the properties
        of the ReadoutAmplifier.  Additional arguments can be provided
        to account for the number of desired overscan transfers The
        result is a simulated final segment image, in ADU.

        Args:
            segment (SegmentSimulator): Simulated segment image to process.
            serial_register (SerialRegister): Serial register to use during
                readout.
            num_serial_overscan (int): Number of serial overscan pixels.
            num_parallel_overscan (int): Number of parallel overscan pixels.

        Returns:
            NumPy array.

        """
        # Create output array
        iy = int(self.ny + parallel_overscan_width)
        ix = int(self.nx + self.prescan_width + serial_overscan_width)
        image = np.random.normal(loc=self.output_amplifier.global_offset,
                                 scale=self.output_amplifier.noise,
                                 size=(iy, ix))
        free_charge = copy.deepcopy(self.segarr)

        # Keyword override toggles
        if kwargs.get('no_trapping', False):
            do_trapping = False
        else:
            do_trapping = self.do_trapping
        if kwargs.get('no_local_offset', False):
            do_local_offset = False
        else:
            do_local_offset = self.output_amplifier.do_local_offset
        if kwargs.get('no_cti', False):
            cti = 0.0
        else:
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
            if do_local_offset:
                offset = self.output_amplifier.local_offset(offset,
                                                            transferred_charge[:, 0])
                image[:iy-parallel_overscan_width, i] += transferred_charge[:, 0] + offset
            else:
                image[:iy-parallel_overscan_width, i] += transferred_charge[:, 0]
            free_charge = np.pad(transferred_charge, ((0, 0), (0, 1)),
                                 mode='constant')[:, 1:] + deferred_charge

            # Trap emission
            if do_trapping:
                for trap in self.serial_traps:
                    released_charge = trap.release_charge()
                    free_charge += released_charge

        return image/float(self.output_amplifier.gain)


class BaseOutputAmplifier:

    do_local_offset = False

    def __init__(self, gain, noise=0.0, global_offset=0.0):

        self.gain = gain
        self.noise = noise
        self.global_offset = global_offset


class FloatingOutputAmplifier(BaseOutputAmplifier):
    """Object representing the readout amplifier of a single channel.

    Attributes:
        noise (float): Value of read noise [e-].
        offset (float): Bias offset level [e-].
        gain (float): Value of amplifier gain [e-/ADU].
        do_bias_drift (bool): Specifies inclusion of bias drift.
        drift_size (float): Strength of bias drift exponential.
        drift_tau (float): Decay time constant for bias drift.
    """
    do_local_offset = True

    def __init__(self, gain, scale, decay_time, noise=0.0, offset=0.0):

        super().__init__(gain, noise, offset)
        self.update_parameters(scale, decay_time)

    def local_offset(self, old, signal):
        """Calculate local offset hysteresis."""

        new = self.scale*signal

        return np.maximum(new, old*np.exp(-1/self.decay_time))

    def update_parameters(self, scale, decay_time):
        """Update parameter values, if within acceptable values."""

        if scale < 0.0:
            raise ValueError("Scale must be greater than or equal to 0.")
        self.scale = scale
        if decay_time <= 0.0:
            raise ValueError("Decay time must be greater than 0.")
        if np.isnan(decay_time):
            raise ValueError("Decay time must be real-valued number, not NaN.")
        self.decay_time = decay_time


class FloatingOutputAmplifier2(BaseOutputAmplifier):
    """Object representing the readout amplifier of a single channel.

    Attributes:
        noise (float): Value of read noise [e-].
        offset (float): Bias offset level [e-].
        gain (float): Value of amplifier gain [e-/ADU].
        do_bias_drift (bool): Specifies inclusion of bias drift.
        drift_size (float): Strength of bias drift exponential.
        drift_tau (float): Decay time constant for bias drift.
    """
    do_local_offset = True

    def __init__(self, gain, scale, decay_time, beta, noise=0.0, offset=0.0):

        super().__init__(gain, noise, offset)
        self.update_parameters(scale, decay_time, beta)

    def local_offset(self, old, signal):
        """Calculate local offset hysteresis."""

        new = self.scale*(signal**self.beta)

        return np.maximum(new, old*np.exp(-1/self.decay_time))

    def update_parameters(self, scale, decay_time, beta):
        """Update parameter values, if within acceptable values."""

        if scale < 0.0:
            raise ValueError("Scale must be greater than or equal to 0.")
        self.scale = scale
        if decay_time <= 0.0:
            raise ValueError("Decay time must be greater than 0.")
        if np.isnan(decay_time):
            raise ValueError("Decay time must be real-valued number, not NaN.")
        self.decay_time = decay_time
        if beta <= 0.0:
            raise ValueError("Beta must be greater than 0.")
        self.beta = beta
