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
__all__ = ("main", )


import argparse
import sys
from lsst.cp.pipe.plotPtc import PlotPhotonTransferCurveTask


def build_argparser():
    """Construct an argument parser for the ``plot_ptc_dataset.py`` script.
    Returns
    -------
    argparser : `argparse.ArgumentParser`
        The argument parser that defines the ``plot_ptc_dataset.py``
        command-line interface.
    """
    parser = argparse.ArgumentParser(description='Rewrite native FITS files from the test '
                                                 'stand to a standard format')
    parser.add_argument('datasetFilename', help="datasetPtc (lsst.ip.isr.PhotonTransferCurveDataset) file"
                        "name (fits)", type=str)
    parser.add_argument('--linearizerFileName', help="linearizer (isr.linearize.Linearizer) file"
                        "name (fits)", type=str, default=None)
    parser.add_argument('--outDir', type=str,
                        help="Root directory to which to write outputs", default='.')
    parser.add_argument('--detNum', type=int,
                        help="Detector number",
                        default=999)
    parser.add_argument('--signalElectronsRelativeA', type=float,
                        help="Signal value for relative systematic bias between different"
                        "methods of estimating a_ij(Fig. 15 of Astier+19)",
                        default=75000)
    parser.add_argument('--plotNormalizedCovariancesNumberOfBins', type=int,
                        help="Number of bins in `plotNormalizedCovariancesNumber` function "
                        "(Fig. 8, 10., of Astier+19)",
                        default=10)

    return parser


def main():
    args = build_argparser().parse_args()
    try:
        plotPtc = PlotPhotonTransferCurveTask(
            args.datasetFilename,
            linearizerFileName=args.linearizerFileName,
            outDir=args.outDir, detNum=args.detNum,
            signalElectronsRelativeA=args.signalElectronsRelativeA,
            plotNormalizedCovariancesNumberOfBins=args.plotNormalizedCovariancesNumberOfBins)

        plotPtc.runDataRef()
    except Exception as e:
        print(f"{e}", file=sys.stderr)
        return 1
    return 0
