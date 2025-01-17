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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

__all__ = ["CpQuadNotchExtractConfig", "CpQuadNotchExtractTask",
           "CpQuadNotchExtractConfig", "CpQuadNotchExtractTask"]

class CpQuadNotchExtractConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "exposure", "detector")):
    inputExp = cT.Input(
        name="cpQnIsrExp",
        doc="Input ISR-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
        multiple=False,
        deferLoad=False,
    )
    outputData = cT.Output(
        name="cpQuadNotchSingle",
        doc="Output quad-notch analysis.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
    )


class CpQuadNotchExtractConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=CpQuadNotchExtractConnections):
    """Configuration for quad-notch processing."""
    nSigma = pexConfig.Field(
        dtype=float,
        default=2.0,
        doc="",
    )
    nPixMin = pexConfig.Field(
        dtype=int,
        default=0,
        doc="",
    )
    grow = pexConfig.Field(
        dtype=int,
        default=0,
        doc="",
    )
    xWindow = pexConfig.Field(
        dtype=int,
        default=0,
        doc="",
    )
    yWindow = pexConfig.Field(
        dtype=int,
        default=50,
        doc="",
    )
    xGauge = pexConfig.Field(
        dtype=float,
        default=1.75,
        doc="",
    )
    threshold = pexConfig.Field(
        dtype=float,
        default=1.2e5,
        doc="",
    )


class CpQuadNotchExtractTask(pipeBase.PipelineTask):
    """Task to measure quad-notch data."""

    ConfigClass = CpQuadNotchExtract
    _DefaultName = "cpQuadNotchExtract"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, inputExp, camera):
        """
        """


        return pipeBase.Struct(
            outputData=outputTable,
        )


class CpQuadNotchMergeConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("instrument", "detector")):
    inputData = cT.Input(
        name="cpQuadNotchSingle",
        doc="Quad-notch measurements from individual exposures.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True
        deferLoad=False,
    )
    outputData = cT.Output(
        name="cpQuadNotch",
        doc="Output combined quad-notch analysis.",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "detector"),
    )


class CpQuadNotchMergeConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=CpQuadNotchMergeConnections):
    """Configuration for quad-notch processing."""
    nSigma = pexConfig.Field(
        dtype=float,
        default=2.0,
        doc="",
    )
    nPixMin = pexConfig.Field(
        dtype=int,
        default=0,
        doc="",
    )
    grow = pexConfig.Field(
        dtype=int,
        default=0,
        doc="",
    )
    xWindow = pexConfig.Field(
        dtype=int,
        default=0,
        doc="",
    )
    yWindow = pexConfig.Field(
        dtype=int,
        default=50,
        doc="",
    )
    xGauge = pexConfig.Field(
        dtype=float,
        default=1.75,
        doc="",
    )
    threshold = pexConfig.Field(
        dtype=float,
        default=1.2e5,
        doc="",
    )


class CpQuadNotchMergeTask(pipeBase.PipelineTask):
    """Task to measure quad-notch data."""

    ConfigClass = CpQuadNotchMerge
    _DefaultName = "cpQuadNotchMerge"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, inputExp, camera):
        """
        """


        return pipeBase.Struct(
            outputData=outputTable,
        )

    
