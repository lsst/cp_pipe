__all__ = ["configureIsrTaskLSSTForCalibrations"]

import numpy as np


def configureIsrTaskLSSTForCalibrations(config):
    """Apply configuration overrides for a baseline of building calibrations.

    Parameters
    ----------
    config : `lsst.ip.isr.IsrTaskLSSTConfig`
        Configuration object to override.
    """
    # These are defined in application/run order.
    config.doBootstrap = False
    config.bssVoltageMinimum = 0.0
    config.doCheckUnprocessableData = False
    config.doDiffNonLinearCorrection = False
    config.doCorrectGains = False
    config.serialOverscanMedianShiftSigmaThreshold = np.inf
    config.ampNoiseThreshold = np.inf
    config.doSaturation = False
    config.doSuspect = False
    config.doApplyGains = False
    config.doCrosstalk = False
    config.crosstalk.doQuadraticCrosstalkCorrection = False
    config.doLinearize = False
    config.doDeferredCharge = False
    config.doAssembleCcd = True
    config.expectWcs = False
    config.doITLSatSagMask = False
    config.doITLEdgeBleedMask = False
    config.doITLDipMask = False
    config.doBias = False
    config.doDark = False
    config.doDefect = False
    config.doNanMasking = True
    config.doWidenSaturationTrails = False
    config.doBrighterFatter = False
    config.doVariance = True
    config.maskNegativeVariance = False
    config.doFlat = False
    config.doSaveInterpPixels = False
    config.doSetBadRegions = False
    config.doInterpolate = False
    config.doAmpOffset = False
    config.doStandardStatistics = True
    config.doCalculateStatistics = True
    config.doBinnedExposures = False
