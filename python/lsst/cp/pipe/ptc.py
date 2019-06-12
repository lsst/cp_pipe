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

__all__ = ['MeasurePhotonTransferCurveTask',
           'MeasurePhotonTransferCurveTaskConfig', ]

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# import lsstDebug
# import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.log as lsstLog
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
#from lsst.atmospec.utils import gainFromFlatPair
from lsst.ip.isr import IsrTask
from .utils import NonexistentDatasetTaskDataIdContainer, PairedVisitListTaskRunner, checkExpLengthEqual, \
    validateIsrConfig


class MeasurePhotonTransferCurveTaskConfig(pexConfig.Config):
    """Config class for photon transfer curve measurement task"""

    isr = pexConfig.ConfigurableField(
        target=IsrTask,
        doc="""Task to perform instrumental signature removal""",
    )
    isrMandatorySteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must be performed for valid results. Raises if any of these are False",
        default=['doAssembleCcd']
    )
    isrForbiddenSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that must NOT be performed for valid results. Raises if any of these are True",
        default=['doFlat', 'doFringe', 'doAddDistortionModel', 'doBrighterFatter', 'doUseOpticsTransmission',
                 'doUseFilterTransmission', 'doUseSensorTransmission', 'doUseAtmosphereTransmission']
    )
    isrDesirableSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that it is advisable to perform, but are not mission-critical." +
        " WARNs are logged for any of these found to be False.",
        default=['doBias', 'doDark', 'doCrosstalk', 'doDefect']
    )
    isrUndesirableSteps = pexConfig.ListField(
        dtype=str,
        doc="isr operations that it is *not* advisable to perform in the general case, but are not" +
        " forbidden as some use-cases might warrant them." +
        " WARNs are logged for any of these found to be True.",
        default=['doLinearize']
    )
    ccdKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to pull a detector from a dataId, e.g. 'ccd' or 'detector'",
        default='ccd',
    )
    imageTypeKey = pexConfig.Field(
        dtype=str,
        doc="The key by which to check whether images are darks or flats",
        default='imgType',
    )
    makePlots = pexConfig.Field(
        dtype=bool,
        doc="Plot the PTC curves?",
        default=False,
    )


class MeasurePhotonTransferCurveTask(pipeBase.CmdLineTask):
    """XXX docstring here
    """

    RunnerClass = PairedVisitListTaskRunner
    ConfigClass = MeasurePhotonTransferCurveTaskConfig
    _DefaultName = "measurePhotonTransferCurve"

    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        # overrideConfig = IsrTask.ConfigClass()
        self.makeSubtask("isr")
        # overrideConfig.assembleCcd.config.doTrim = False
        # self.isr.applyOverrides(overrideConfig)
        # self.isr.assembleCcd.config.doTrim = False
        self.isr.log.setLevel(lsstLog.WARN)  # xxx consider this level

        plt.interactive(False)  # stop windows popping up when plotting. When headless, use 'agg' backend too
        validateIsrConfig(self.isr, self.config.isrMandatorySteps,
                          self.config.isrForbiddenSteps, self.config.isrDesirableSteps, checkTrim=False)
        self.config.validate()
        self.config.freeze()

    @classmethod
    def _makeArgumentParser(cls):
        """Augment argument parser for the MeasurePhotonTransferCurveTask."""
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--visit-pairs", dest="visitPairs", nargs="*",
                            help="Visit pairs to use. Each pair must be of the form INT,INT e.g. 123,456")
        parser.add_id_argument("--id", datasetType="measurePhotonTransferCurveDataset",
                               ContainerClass=NonexistentDatasetTaskDataIdContainer,
                               help="The ccds to use, e.g. --id ccd=0..100")
        return parser

    @pipeBase.timeMethod
    def runDataRef(self, dataRef, visitPairs):
        """Run the PTC measurement task.

        For a dataRef (which is each detector here),
        and given a list of visit pairs, measure
        the photon transfer curve.

        Parameters
        ----------
        dataRef : list of lsst.daf.persistence.ButlerDataRef
            dataRef for the detector for the visits to be fit.
        visitPairs : `iterable` of `tuple` of `int`
            Pairs of visit numbers to be processed together
        """

        # setup necessary objects
        detNum = dataRef.dataId[self.config.ccdKey]
        detector = dataRef.get('camera')[dataRef.dataId[self.config.ccdKey]]
        ampInfoCat = detector.getAmpInfoCatalog()
        ampNames = [amp.getName() for amp in ampInfoCat]
        dataDict = {key: {} for key in ampNames}

        self.log.info('Measuring PTC using %s visits for detector %s' % (visitPairs, detNum))

        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(5)
        sctrl.setNumIter(3)
        statTypes = afwMath.MEANCLIP | afwMath.STDEVCLIP

        for (v1, v2) in visitPairs:  # xxx change this to use the pairs!
            dataRef.dataId['visit'] = v1
            exp1 = self.isr.runDataRef(dataRef).exposure
            # dataRef.dataId['visit'] = v2
            # exp2 = self.isr.runDataRef(dataRef).exposure
            # del dataRef.dataId['visit']

            # raw = dataRef.get('raw', visit=v1)
            # exp2 = dataRef.get('raw', visit=v2)
            # checkExpLengthEqual(exp1, exp2, v1, v2, raiseWithMessage=True)
            # gains = gainFromFlatPair(exp1, exp2, 'simple', rawExpForNoiseCalc=exp1)

            for amp in detector:
                imArea = exp1.maskedImage[amp.getBBox()]
                stats = afwMath.makeStatistics(imArea, statTypes, sctrl)

                std, stderr = stats.getResult(afwMath.STDEVCLIP)
                mean, meanerr = stats.getResult(afwMath.MEANCLIP)
                npMean = np.mean(exp1.maskedImage[amp.getBBox()].image.array)
                npStd = np.std(exp1.maskedImage[amp.getBBox()].image.array)

                data = dict(npMean=npMean, npStd=npStd, meanClip=mean, stdClip=std)

                ampName = amp.getName()
                dataDict[ampName][v1] = data

        self.log.info(f'Writing PTC data to {dataRef.getUri(write=True)}')
        dataRef.put(dataDict)

        if self.config.makePlots:
            self.plot(dataRef, dataDict)

        self.log.info('Finished measuring PTC for in detector %s' % detNum)
        return pipeBase.Struct(exitStatus=0)

    def plot(self, dataRef, data):
        dirname = dataRef.getUri(datasetType='cpPipePlotRoot', write=True)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detNum = dataRef.dataId[self.config.ccdKey]
        filename = f"PTC_det{detNum}.pdf"
        filenameFull = os.path.join(dirname, filename)
        with PdfPages(filenameFull) as pdfPages:
            self._plotPTC(data)
            pdfPages.savefig()

    def _plotPtc(self, data):
        return
