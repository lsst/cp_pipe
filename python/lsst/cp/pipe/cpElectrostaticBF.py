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
"""Computation of electrostatic solution of brighter-fatter effect impact
on pixel distortions"""

__all__ = ['ElectrostaticBrighterFatterSolveTask',
           'ElectrostaticBrighterFatterSolveConfig']

import numpy as np
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.afw.cameraGeom import FOCAL_PLANE

from .utils import (
    extractCalibDate,
    ElectrostaticFit,
)
from .cpLinearitySolve import ptcLookup
from lsst.ip.isr.isrFunctions import symmetrize
from lsst.ip.isr import ElectrostaticBrighterFatterDistortionMatrix
from lmfit import Parameters, report_fit

# Physical constants for the 
# Following Rajkanan et al. (1979)
BETA = 7.021e-4  # eV K^-1
GAMMA = 1108.0  # K
E_g0 = [1.1557, 2.5]  # eV
E_gd0 = 3.2  # eV
E_p = [1.827e-2, 5.773e-2]  # eV
C = [5.5, 4.0]  # unitless
A = [3.231e2, 7.237e3]  # cm^-1 eV^-2
A_d = 1.052e6  # cm^-1 eV^-2
H_BAR = 6.582e-16  # eV s
K_B = 8.617e-5  # eV K^-1

def lookupStaticCalibrations(datasetType, registry, quantumDataId, collections):
    # For static calibrations, we search with a timespan that has unbounded
    # begin and end; we'll get an error if there's more than one match (because
    # then it's not static).
    timespan = Timespan(begin=None, end=None)
    result = []
    # First iterate over all of the data IDs for this dataset type that are
    # consistent with the quantum data ID.
    for dataId in registry.queryDataIds(datasetType.dimensions, dataId=quantumDataId):
        # Find the dataset with this data ID using the unbounded timespan.
        if ref := registry.findDataset(datasetType, dataId, collections=collections, timespan=timespan):
            result.append(ref)
    return result


class PhotonConversionDepthProbabilityModel():
    """
    Class to compute the probability distribution of photon conversion depths.
    """
    def __init__(self, detector, transmission_filter_detector):
        """
        Initialize the photon conversion depth probability model.

        Parameters
        ----------
        detector : `lsst.afw.cameraGeom.Detector`
            Detector associated with the transmission_filter_detector.
        transmission_filter_detector : `lsst.ip.isr.TransmissionFilterDetector`
            Transmission filter detector to use for the probability distribution.
        """
        self.filter = transmission_filter_detector
        self.detector = detector
    
    def compute(self, depths=None, wavelengths=None, temperature=173.0, flat_sed_weights=True):
        """
        Evaluate the photon conversion depth probability model.

        The probability distribution is computed assuming an incident flat SED 
        on the filter at the location of the detector in the focal plane.

        The probability distribution is computed as:
        p(d) = sum_i w_i * alpha_i * exp(-alpha_i * d)
        where w_i is the weight of the wavelength, alpha_i is the optical absorption 
        coefficient, and d is the depth.

        The probability distribution is computed as:
        p(d) = sum_i w_i * alpha_i * exp(-alpha_i * d)
        where w_i is the weight of the wavelength, alpha_i is the optical absorption 
        coefficient, and d is the depth.

        We assume that the photons convert in depth bins of width depths[i+1] - depths[i].

        Parameters
        ----------
        depths : `np.ndarray`
            Depths to evaluate the probability distribution at (units of um).
            If None, will default to [0, 100] um in steps of 1 um.
        wavelengths : `np.ndarray`
            Wavelengths to evaluate the probability distribution at (units of nm).
            If None, will default to computing in steps of 1 nm over 
            transmission_filter_detector.getWavelengthBounds().
        temperature : float, optional
            Temperature in Kelvin of the silicon detector (default is 173.0 K for
            the LSST Camera).
        flat_sed_weights : bool, optional
            If True (default), weight by throughput * wavelength for a flat F_λ
            SED (photon count ∝ λ). If False, weight by throughput only.

        Returns
        -------
        (d, p) : tuple, (np.ndarray, np.ndarray)
            Tuple of arrays containing the probability p ([0,1]) of a photon 
            converting at d (midpoint) between (depths[i], depths[i+1]) and 
            the depth bin midpoints (units of um), assuming an incident flat 
            SED incident on the filter at the location of the detector in the 
            focal plane. The model MAY NOT be normalized to 1.
        """
        # Default depths
        if depths is None:
            depths = np.linspace(0.0, 100.0, int(100+1))
        else:
            depths = np.asarray(depths)

        # Bin midpoints and edges
        depth_bin_midpoints = (depths[:-1] + depths[1:]) / 2.0

        # Default wavelengths
        if wavelengths is None:
            lambda_min, lambda_max = self.filter.getWavelengthBounds() # in Angstroms
            lambda_min = lambda_min / 10.0 # in nm
            lambda_max = lambda_max / 10.0 # in nm
            n_samples = int(lambda_max-lambda_min+1)
            wavelengths = np.linspace(lambda_min, lambda_max, n_samples)
        else:
            wavelengths = np.asarray(wavelengths)


        # The filter transmission is a function of wavelength and position at the location
        # of thed etector in the focal plane. Get the detector center:
        detectorCenter = detector.getCenter(FOCAL_PLANE)

        throughput = self.filter.sampleAt(position=detectorCenter, wavelengths=wavelengths)
        if flat_sed_weights:
            weight = throughput * wavelengths
        else:
            weight = throughput
        weight = weight / np.maximum(weight.sum(), np.finfo(float).tiny)

        # Compute the optical absorption coefficient as a function of wavelength 
        # and temperature.
        alpha = rajkanan_1979_alpha(T=temperature, wavelength=wavelengths)  # (n_wavelen,) in um^-1
        
        # Optional: 
        # Compute the PDF(d) = sum_i w_i * alpha_i * exp(-alpha_i * d)
        # log_alpha = np.where(alpha > 0, np.log(alpha), -np.inf)
        # log_pdf = log_alpha[:, np.newaxis] - alpha[:, np.newaxis] * depth_um[np.newaxis, :]
        # pdf = np.exp(np.log(weight[:, np.newaxis]) + log_pdf)
        # pdf = np.where(alpha[:, np.newaxis] > 0, pdf, 0.0)
        # p = pdf.sum(axis=0)
        # norm = np.trapezoid(p, depth_um)
        # p = p / np.maximum(norm, np.finfo(float).tiny)
        # result[f] = (p, depth_um)

        # CDF(d) = sum_i w_i * (1 - exp(-alpha_i * d)); p_i = CDF(edge[i+1]) - CDF(edge[i])
        cdf_at_edges = (weight[:, np.newaxis] * (1.0 - np.exp(-alpha[:, np.newaxis] * depths[np.newaxis, :]))).sum(axis=0)
        p = np.diff(cdf_at_edges)
        conversion_weights = (depth_bin_midpoints, p)

        # NOTE: the probability distribution should be normalized to 1.0, 
        # important for flux conservation, and may not be true at this 
        # point. However, this is ensured when it is used in 
        # electrostaticFit.computePixelDistortions()

        return conversion_weights


    # Utilities
    def E_g(idx, T):
        """
        Indirect band gap energy for silicon (Rajkanan et al. (1979), Eq. 3).

        Parameters
        ----------
        idx : int
            Band index: 0 for the lower indirect gap (~1.16 eV), 1 for the higher
            indirect gap (~2.5 eV).
        T : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Indirect band gap in eV.
        """
        return E_g0[idx] - ((BETA*(T**2)) / (T + GAMMA))
        
    def E_gd(T):
        """
        Direct band gap energy for silicon (Rajkanan et al. (1979), Eq. 3).

        Parameters
        ----------
        T : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Direct band gap in eV.
        """
        return E_gd0 - ((BETA*(T**2)) / (T + GAMMA))

    def wavelength_to_frequency(wavelength):
        """
        Convert photon wavelength to angular frequency.

        Parameters
        ----------
        wavelength : float or np.ndarray
            Photon wavelength in nanometers.

        Returns
        -------
        float or np.ndarray
            Angular frequency ω in rad/s. Same shape as input.

        Notes
        -----
        Uses ω = 2πc/λ so that ℏω gives photon energy in eV when used with
        the module constant H_BAR.
        """
        # Speed of light in nm/s
        c = 2.99792458e17  # nm/s
        
        # Convert wavelength to angular frequency
        # ω = 2π * ν = 2π * c/λ
        omega = 2 * np.pi * c / wavelength
        
        return omega


    def rajkanan_1979_alpha(T, wavelength):
        """
        Optical absorption coefficient of silicon as a function of wavelength
        and temperature (from Rajkanan et al. (1979), Eq. 4).

        Models indirect (phonon-assisted) and direct transitions. Valid for
        photon energies roughly 1.1–4.0 eV and temperatures 20–500 K.

        Parameters
        ----------
        T : float
            Temperature in Kelvin.
        wavelength : float or array-like
            Photon wavelength(s) in nanometers.

        Returns
        -------
        np.ndarray
            Absorption coefficient in um⁻¹. Same shape as wavelength (0-d for
            scalar input).

        Notes
        -----
        Indirect terms are included only when photon energy exceeds the
        relevant threshold (E_g ± E_p). The direct term is included only
        when photon energy exceeds the direct gap E_gd.

        References
        ----------
        .. [1] Rajkanan, K., Singh, R., & Shewchun, J. (1979). Absorption
            coefficient of silicon for solar cell calculations. Solid
            State Electronics, 22(9), 793-795.
        """
        wavelength = np.asarray(wavelength)
        omega = wavelength_to_frequency(wavelength)

        sum_ij = 0.0
        for i in range(2):
            for j in range(2):
                c1 = C[i]*A[j]
                # Indirect terms: only when photon energy is above threshold for that process.
                # Phonon absorption: ℏω ≥ E_g - E_p; phonon emission: ℏω ≥ E_g + E_p.
                # Otherwise squaring (ℏω - E_g ± E_p) when negative gives unphysical rise at long λ.
                hw = H_BAR * omega
                Eg_j = E_g(j, T)
                term_abs = np.maximum(0.0, hw - Eg_j + E_p[i])**2 / (np.exp(E_p[i]/(K_B*T)) - 1)
                term_em = np.maximum(0.0, hw - Eg_j - E_p[i])**2 / (1 - np.exp(-E_p[i]/(K_B*T)))
                c2 = term_abs + term_em
                sum_ij += c1 * c2

        # Direct-gap term: only real when photon energy exceeds E_gd
        excess = H_BAR * omega - E_gd(T)
        c = A_d * (np.maximum(0.0, excess))**(1.0 / 2.0)

        alpha = sum_ij + c  # cm^-1
        alpha_per_m = alpha * 1e-4  # um^-1
        return np.asarray(alpha_per_m)


class ElectrostaticBrighterFatterSolveConnections(pipeBase.PipelineTaskConnections,
                                                  dimensions=("instrument", "detector")):
    dummy = cT.Input(
        name="raw",
        doc="Dummy exposure.",
        storageClass='Exposure',
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
        deferLoad=True,
    )
    camera = cT.PrerequisiteInput(
        name="camera",
        doc="Camera associated with this data.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    inputPtc = cT.PrerequisiteInput(
        name="ptc",
        doc="Photon transfer curve dataset.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        lookupFunction=ptcLookup,
    )
    inputBfPtc = cT.Input(
        name="bfPtc",
        doc="Input BF PTC dataset.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    transmission_filter_detector = connectionTypes.PrerequisiteInput(
        doc="Filter transmission curve information",
        name="transmission_filter_detector",
        storageClass="TransmissionCurve",
        dimensions=("band", "instrument", "physical_filter", "detector"),
        lookupFunction=lookupStaticCalibrations,
        isCalibration=True,
        deferLoad=True,
        multiple=True,
    )

    output = cT.Output(
        name="electroBfDistortionMatrix",
        doc="Output measured brighter-fatter electrostatic model.",
        storageClass="IsrCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        if config.useBfPtc:
            del self.inputPtc
            del self.dummy
        else:
            del self.inputBfPtc
        if config.doColorCorrection:



class ElectrostaticBrighterFatterSolveConfig(pipeBase.PipelineTaskConfig,
                                             pipelineConnections=ElectrostaticBrighterFatterSolveConnections):

    useBfPtc = pexConfig.Field(
        dtype=bool,
        doc="Use a BF ptc in a single pipeline?",
        default=False,
    )
    fitRange = pexConfig.Field(
        dtype=int,
        doc="Maximum pixel range to compute the electrostatic fit.",
        default=8,
    )
    fitMethod = pexConfig.Field(
        dtype=str,
        doc="Minimization technique to fit the electrostatic solution. "
            "Should be one of the available fitting methods in "
            "`lmfit.minimizer.Minimizer.minimize`. For list of all possible "
            "methods see the documentation.",
        default="leastsq",
    )
    doNormalizeElectrostaticModel = pexConfig.Field(
        dtype=bool,
        doc="Do you want apply a final normalization to the modeled "
            "aMatrix?",
        default=False,
    )
    doFitNormalizationOffset = pexConfig.Field(
        dtype=bool,
        doc="Do you want to fit an offset to the a matrix? This caused "
            "by long range correlations in the data. Only used if "
            "doNormalizeElectrostaticModel.",
        default=True,
    )
    nImageChargePairs = pexConfig.Field(
        dtype=int,
        doc="Number of image charge pairs to use when computing "
            "Gauss's law. The larger number, the better, and an "
            "odd number is preferred.",
        default=11,
    )
    initialParametersDict = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc="Initial fit parameters, should contain `thickness`, "
            "`pixelsize`, `zq`, zsh`, `zsv`, `a`, `b`, `alpha`, "
            " and `BETA`. See the class docstring for descriptions "
            " and units of each parameter.",
        default={
            'thickness': 100.0,
            'pixelsize': 10.0,
            'zq': 1.0,
            'zsh': 2.0,
            'zsv': 3.0,
            'a': 2.0,
            'b': 2.0,
            'alpha': 1.0,
            'BETA': 0.0,
        },
    )
    parametersToVary = pexConfig.DictField(
        keytype=str,
        itemtype=bool,
        doc="Dictionary of parameters and booleans which will configure "
            "if the parameter is allowed to vary in the fit, should contain "
            "`thickness`,`pixelsize`, `zq`, zsh`, `zsv`, `a`, `b`, `alpha`, "
            "and `BETA`. If False, the parameter will be fixed to the initial "
            "value set in initialParameterDict. See the class docstring for "
            "descriptions and units of each parameter.",
        default={
            'thickness': False,
            'pixelsize': False,
            'zq': True,
            'zsh': True,
            'zsv': True,
            'a': True,
            'b': True,
            'alpha': True,
            'BETA': True,
        },
    )
    doColorCorrection = pexConfig.Field(
        dtype=bool,
        doc="Do you want to include a conversion depth distribution in the "
            "electrostatic fit and generate an electrostatic distortion "
            "matrix per filter? If False, will assume all photons convert "
            "at the surface of the detector. If True, it will assume a flat "
            "SED incident on a filter to compute the conversion depth "
            "distribution.",
        default=True,
    )
    physicalFiltersToSolve = pexConfig.ListField(
        dtype=str,
        doc="List of physical filters in which to compute the pixel boundary 
            "distortions. Only used if doColorCorrection is True.",
        default=["u_24", "g_6", "r_57", "i_39", "z_20", "y_10"],
    )


class ElectrostaticBrighterFatterSolveTask(pipeBase.PipelineTask):
    """Find the complete electrostatic solution to the given PTC.
    """

    ConfigClass = ElectrostaticBrighterFatterSolveConfig
    _DefaultName = 'cpElectrostaticBfSolve'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """Ensure that the input and output dimensions are passed along.

        Parameters
        ----------
        butlerQC : `lsst.daf.butler.QuantumContext`
            Butler to operate on.
        inputRefs : `lsst.pipe.base.InputQuantizedConnection`
            Input data refs to load.
        ouptutRefs : `lsst.pipe.base.OutputQuantizedConnection`
            Output data refs to persist.
        """
        inputs = butlerQC.get(inputRefs)

        # Use the dimensions to set
        # electroBfDistortionMatrix/provenance
        # information.
        if self.config.useBfPtc:
            inputs["inputDims"] = dict(inputRefs.inputBfPtc.dataId.required)
            inputs["inputPtc"] = inputs["inputBfPtc"]
            inputs["dummy"] = []
            del inputs["inputBfPtc"]
        else:
            inputs["inputDims"] = dict(inputRefs.inputPtc.dataId.required)

        # Add calibration provenance info to header.
        kwargs = dict()
        reference = getattr(inputRefs, "inputPtc", None)

        if reference is not None and hasattr(reference, "run"):
            runKey = "PTC_RUN"
            runValue = reference.run
            idKey = "PTC_UUID"
            idValue = str(reference.id)
            dateKey = "PTC_DATE"
            calib = inputs.get("inputPtc", None)
            dateValue = extractCalibDate(calib)

            kwargs[runKey] = runValue
            kwargs[idKey] = idValue
            kwargs[dateKey] = dateValue

            self.log.info("Using " + str(reference.run))

        outputs = self.run(**inputs)
        outputs.output.updateMetadata(setDate=False, **kwargs)

        butlerQC.put(outputs, outputRefs)

    def run(self, inputPtc, dummy, camera, inputDims):
        """Fit the PTC A MATRIX into a vectorized a matrix form
        based on a complete electrostatic solution.

        Parameters
        ----------
        inputPtc : `lsst.ip.isr.PhotonTransferCurveDataset`
            PTC data containing per-amplifier covariance measurements.
        dummy : `lsst.afw.image.Exposure`
            The exposure used to select the appropriate PTC dataset.
            In almost all circumstances, one of the input exposures
            used to generate the PTC dataset is the best option.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera to use for camera geometry information.
        inputDims : `lsst.daf.butler.DataCoordinate` or `dict`
            DataIds to use to populate the output calibration.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The resulst struct containing:

            ``output``
                Resulting Brighter-Fatter electrostatic solution
                (`lsst.ip.isr.ElectrostaticBrighterFatterDistortionMatrix`).
        """
        detector = camera[inputDims['detector']]

        inputRange = int(inputPtc.covMatrixSideFullCovFit)
        fitRange = int(self.config.fitRange)

        # Check if the PTC fit type is valid given the configuration
        if not inputPtc.ptcFitType.startswith("FULLCOVARIANCE"):
            raise ValueError(
                "ptcFitType must be FULLCOVARIANCE* to solve for electrostatic solution."
            )
        if fitRange > inputRange:
            raise ValueError(
                "Cannot compute the electrostatic solution if "
                "int(inputPtc.covMatrixSide) < fitRange."
            )
        if (inputPtc.metadata['FILTER'] is None) and self.config.doColorCorrection:
            # Do not know the filter name of the input flats to the PTC
            # Check if a dummy exposure is there, which will contain the filter name
            if dummy is not None and dummy.metadata['FILTER'] is not None:
                # Get the transmission for this filter
            else:
                # We don't know what filter is associated with the PTC
                self.log.warning("No filter name found in the PTC metadata and no dummy exposure. "
                                 "Default is to solve initial electrostatic model assuming all "
                                 "photons convert at the surface of the detector.")
                


        
        # Get transmission curves for the filters to solve
        if self.config.doColorCorrection:
            availableFilters = list(self.config.physicalFiltersToSolve)

            for physicalFilter in self.config.physicalFiltersToSolve:
                try:
                    transmission_filter_detector = self.get(self.config.transmission_filter_detector)
                except LookupError:
                    self.log.warning(
                        "No transmission curves found for the filters to solve. "
                        "Please ensure that the transmission curves are available "
                        "in the database."
                    )
                    availableFilters.remove(physicalFilter)

            if len(availableFilters) == 0:
                self.log.warning("No transmission curves found for the filters to solve. "
                                 "Please ensure that the transmission curves are available "
                                 "in the database.")
        else:
            availableFilters = []

        # Initialize the output calibration
        electroBfDistortionMatrix = ElectrostaticBrighterFatterDistortionMatrix(
            camera=camera,
            detectorId=detector.getId(),
            inputRange=inputRange,
            fitRange=fitRange,
            availableFilters=availableFilters,
        )

        # Inherit data + metadata
        electroBfDistortionMatrix.updateMetadataFromExposures([inputPtc])

        badAmps = inputPtc.badAmps
        electroBfDistortionMatrix.badAmps = badAmps
        electroBfDistortionMatrix.gain = inputPtc.gain

        aMatrixDict = inputPtc.aMatrix
        aMatrixList = np.array([m for _, m in aMatrixDict.items() if _ not in badAmps])

        nGoodAmps = len(detector.getAmplifiers()) - len(badAmps)
        if nGoodAmps == 0:
            self.log.warning("The entire detector is bad and cannot generate a "
                             "detector solution.")
            return pipeBase.Struct(
                output=electroBfDistortionMatrix,
            )
        elif nGoodAmps < 2:
            # If the general uncertainty is one, the measurement
            # uncertainties along the axes are sqrt(2), and sqrt(8)
            # in (0,0) (because the slope of C00 is fitted).
            #
            # This sets variances at (1, 2, 8) for the three groups.
            # Then the number of replicas (when going to 4 quadrants)
            # are (4, 2, 1) for the same three groups.
            #
            # The effective variances are then in the ratios (1/4, 1, 8)
            # or (1, 4, 32).
            self.log.warning("Not enough good amplifiers in this detector "
                             "to confidently solve. Setting aMatrixSigma "
                             "to default.")
            aMatrix = np.mean(aMatrixList, axis=0)
            aMatrixSigma = np.ones_like(aMatrix, dtype=np.float64)
            aMatrixSigma[0, :] = 2.0
            aMatrixSigma[:, 0] = 2.0
            aMatrixSigma[0, 0] = np.sqrt(32)

        else:
            # Do a quick sigma-clipped mean
            errors = aMatrixList - np.mean(aMatrixList, axis=0)
            sigmaErrors = errors / np.std(aMatrixList, axis=0)
            badValues = np.argwhere(np.abs(sigmaErrors) > 3)
            aMatrixList[badValues] = np.nan
            aMatrix = np.nanmean(aMatrixList, axis=0)
            aMatrixSigma = np.nanstd(aMatrixList, axis=0)

        # Ensure we have numpy arrays in 64-bit float precision
        aMatrix = np.asarray(aMatrix, dtype=np.float64)
        aMatrixSigma = np.asarray(aMatrixSigma, dtype=np.float64)

        # Set initial parameters using config
        thickness = np.float64(self.config.initialParametersDict['thickness'])
        pixelsize = np.float64(self.config.initialParametersDict['pixelsize'])
        zq = np.float64(self.config.initialParametersDict['zq'])
        zsh = np.float64(self.config.initialParametersDict['zsh'])
        zsv = np.float64(self.config.initialParametersDict['zsv'])
        a = np.float64(self.config.initialParametersDict['a'])
        b = np.float64(self.config.initialParametersDict['b'])
        alpha = np.float64(self.config.initialParametersDict['alpha'])
        BETA = np.float64(self.config.initialParametersDict['BETA'])

        initialParams = Parameters()
        initialParams.add(
            "thickness",
            value=thickness,
            min=0,
            max=1.25*thickness,
            vary=self.config.parametersToVary["thickness"],
        )
        initialParams.add(
            "pixelsize",
            value=pixelsize,
            min=0.5*np.abs(pixelsize),
            max=1.5*np.abs(pixelsize),
            vary=self.config.parametersToVary["pixelsize"],
        )
        initialParams.add(
            "zq",
            value=zq,
            vary=self.config.parametersToVary["zq"],
            min=0.0,
            max=0.5*thickness,
        )
        # These nuisance parameters ensure that
        # (zsh > zq) & (zsv > zq)
        initialParams.add(
            "zsh_minus_zq",
            value=zsh - zq,
            vary=self.config.parametersToVary["zsh"],
            min=1.0e-12,
            max=0.1*thickness,
        )
        initialParams.add(
            "zsh",
            vary=self.config.parametersToVary["zsh"],
            min=0.0,
            max=0.5*thickness,
            expr="zq + zsh_minus_zq" if self.config.parametersToVary["zsh"] else f"{zsh}",
        )
        initialParams.add(
            "zsv_minus_zq",
            value=zsv - zq,
            vary=self.config.parametersToVary["zsv"],
            min=1.0e-12,
            max=0.1*thickness,
        )
        initialParams.add(
            "zsv",
            vary=self.config.parametersToVary["zsv"],
            min=0.0,
            max=0.5*thickness,
            expr="zq + zsv_minus_zq" if self.config.parametersToVary["zsv"] else f"{zsv}",
        )
        initialParams.add(
            "a",
            value=a,
            vary=self.config.parametersToVary["a"],
            min=1.0e-5,
            max=0.35*pixelsize,
        )
        initialParams.add(
            "b",
            value=b,
            vary=self.config.parametersToVary["b"],
            min=1.0e-5,
            max=0.35*pixelsize,
        )
        initialParams.add(
            "alpha",
            value=alpha,
            vary=self.config.parametersToVary["alpha"],
            min=-10.0,
            max=10.0,
        )
        initialParams.add(
            "BETA",
            value=BETA,
            vary=self.config.parametersToVary["BETA"],
            min=-10.0,
            max=10.0,
        )

        # If we want to do color correction, compute the electrostatic fit
        # assuming the color of incident photons in the filter used to construct the PTC
        # follow a flat SED. Otherwise, compute the pixel distortions assuming all photons 
        # convert at the surface of the detector.
        if self.config.doColorCorrection:
            # Compute the conversion depth probability distribution
            conversoinModel = PhotonConversionDepthProbabilityModel(
                detector=detector,
                transmission_filter_detector=transmission_filter_detector,
            )
            conversionWeights = conversoinModel.compute(
                temperature=173.0,
                flat_sed_weights=True,
            )
        else:
            conversionWeights = None

        # Compute the electrostatic fit
        electrostaticFit = ElectrostaticFit(
            initialParams=initialParams,
            fitMethod=self.config.fitMethod,
            aMatrix=aMatrix,
            aMatrixSigma=aMatrixSigma,
            fitRange=fitRange,
            doFitNormalizationOffset=self.config.doFitNormalizationOffset,
            nImageChargePairs=self.config.nImageChargePairs,
            conversionWeights=conversionWeights,
        )

        # Do the fit
        result = electrostaticFit.fit()

        # Check if fit was successful
        if not result.success:
            self.log.warning(
                f"Fit was not successful on first try; loosening tolerances and retrying. {result.message}",
            )
            electrostaticFit = ElectrostaticFit(
                initialParams=initialParams,
                fitMethod=self.config.fitMethod,
                aMatrix=aMatrix,
                aMatrixSigma=aMatrixSigma,
                fitRange=fitRange,
                doFitNormalizationOffset=self.config.doFitNormalizationOffset,
                nImageChargePairs=self.config.nImageChargePairs,
            )

            # Do the fit
            result = electrostaticFit.fit(ftol=1e-7)

            if not result.success:
                raise RuntimeError(f"Re-fit was not successful: {result.message}")

        # Save the fit
        finalParams = result.params
        finalParamsDict = finalParams.valuesdict()

        # No longer need these nusiance variables
        if 'zsh_minus_zq' in finalParamsDict:
            del finalParamsDict['zsh_minus_zq']
        if 'zsv_minus_zq' in finalParamsDict:
            del finalParamsDict['zsv_minus_zq']

        fitParamNames = list(finalParamsDict.keys())
        freeFitParamNames = result.var_names
        electroBfDistortionMatrix.fitParamNames = fitParamNames
        electroBfDistortionMatrix.freeFitParamNames = freeFitParamNames
        electroBfDistortionMatrix.fitParams = finalParamsDict
        fitParamErrors = dict()
        for fitParamName in fitParamNames:
            if fitParamName in freeFitParamNames:
                fitParamErrors[fitParamName] = finalParams[fitParamName].stderr
            else:
                fitParamErrors[fitParamName] = 0.0
        electroBfDistortionMatrix.fitParamErrors = fitParamErrors

        electroBfDistortionMatrix.fitChi2 = result.chisqr
        electroBfDistortionMatrix.fitReducedChi2 = result.redchi
        electroBfDistortionMatrix.fitParamCovMatrix = result.covar

        # Compute the final model
        aMatrixModel = electrostaticFit.model(result.params)

        # Optional:
        # Perform the final model normalization
        modelNormalization = [1.0, 0.0]
        if self.config.doNormalizeElectrostaticModel:
            m, o = electrostaticFit.normalizeModel(aMatrixModel)
            modelNormalization = [m, o]
            aMatrixModel = m*aMatrixModel + o
            self.log.info(
                "Normalization (factor, offset) for amp %s: (%.3f, %.3f)", m, o
            )

        # Save the original data and the final model.
        electroBfDistortionMatrix.aMatrix = aMatrix
        electroBfDistortionMatrix.aMatrixSigma = aMatrixSigma
        electroBfDistortionMatrix.aMatrixModel = aMatrixModel
        electroBfDistortionMatrix.aMatrixSum = symmetrize(aMatrix).sum()
        electroBfDistortionMatrix.aMatrixModelSum = symmetrize(aMatrixModel).sum()
        electroBfDistortionMatrix.modelNormalization = modelNormalization

        # Fit result information
        self.log.info(report_fit(result))

        # Compute the pixel distortions assuming all photons 
        # convert at the surface of the detector
        pd = electrostaticFit.computePixelDistortions(conversionWeights=None)

        aN, aS, aE, aW = (pd.aN, pd.aS, pd.aE, pd.aW)
        ath = pd.ath
        athMinusBeta = pd.athMinusBeta
        usedPixels = np.zeros_like(aN, dtype=bool)
        usedPixels[:fitRange, :fitRange] = True

        electroBfDistortionMatrix.usedPixels = usedPixels
        electroBfDistortionMatrix.ath = ath
        electroBfDistortionMatrix.athMinusBeta = athMinusBeta
        electroBfDistortionMatrix.aN = aN
        electroBfDistortionMatrix.aS = aS
        electroBfDistortionMatrix.aE = aE
        electroBfDistortionMatrix.aW = aW

        # If we want to do color correction, compute the pixel distortions
        # for each filter
        if self.config.doColorCorrection:
            # Populate the pixel distortions for each filter
            for physicalFilter in self.config.physicalFiltersToSolve:
                conversionModel = PhotonConversionDepthProbabilityModel(
                    detector=detector,
                    transmission_filter_detector=physicalFilter,
                )
                conversionWeights = conversionModel.compute(
                    temperature=173.0,
                    flat_sed_weights=True,
                )
                pd = electrostaticFit.computePixelDistortions(conversionWeights=conversionWeights)
                aN, aS, aE, aW = (pd.aN, pd.aS, pd.aE, pd.aW)
                ath = pd.ath
                athMinusBeta = pd.athMinusBeta
                electroBfDistortionMatrix.aNByFilter[physicalFilter] = aN
                electroBfDistortionMatrix.aSByFilter[physicalFilter] = aS
                electroBfDistortionMatrix.aEByFilter[physicalFilter] = aE
                electroBfDistortionMatrix.aWByFilter[physicalFilter] = aW
                electroBfDistortionMatrix.athByFilter[physicalFilter] = ath
                electroBfDistortionMatrix.athMinusBetaByFilter[physicalFilter] = athMinusBeta

        # Optional: Check for validity
        # if self.config.doCheckValidity:
        #     # Todo:
        #     pass

        return pipeBase.Struct(
            output=electroBfDistortionMatrix,
        )

