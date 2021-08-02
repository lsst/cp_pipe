.. program:: plotPhotonTransferCurve.py

##########################
plotPhotonTransferCurve.py
##########################

Construct a series of common plots for the photon transfer curve (PTC) for visualization and quality check.

.. code-block:: text

    usage: plotPhotonTransferCurve.py [-h] [--linearizerFileName LINEARIZERFILENAME] [--outDir OUTDIR]
                                      [--detNum DETNUM] [--signalElectronsRelativeA SIGNALELECTRONSRELATIVEA]
                                      [--plotNormalizedCovariancesNumberOfBins PLOTNORMALIZEDCOVARIANCESNUMBEROFBINS]
                                      datasetFilename

Positional arguments
====================

.. option:: datasetFilename

    datasetPtc (lsst.ip.isr.PhotonTransferCurveDataset) filename (fits)

Optional arguments
==================

.. option::  -h, --help
    Show the help message and exit.

.. option::  --linearizerFileName LINEARIZERFILENAME
    Linearizer (isr.linearize.Linearizer) filename (fits)

.. option::  --outDir OUTDIR
    Root directory to which to write outputs


.. option::  --detNum DETNUM
    Detector number

.. option::  --signalElectronsRelativeA SIGNALELECTRONSRELATIVEA
    Signal value for relative systematic bias between differentmethods of estimating a_ij(Fig. 15 of Astier+19)

.. option::  --plotNormalizedCovariancesNumberOfBins PLOTNORMALIZEDCOVARIANCESNUMBEROFBINS
    Number of bins in `plotNormalizedCovariancesNumber` function (Fig. 8, 10., of Astier+19)
