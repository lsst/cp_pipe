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

__all__ = ['PairedVisitListTaskRunner', 'SingleVisitListTaskRunner',
           'NonexistentDatasetTaskDataIdContainer', 'parseCmdlineNumberString',
           'countMaskedPixels', 'checkExpLengthEqual']

import re
import numpy as np

import lsst.pipe.base as pipeBase
import lsst.ip.isr as ipIsr
import lsst.log as lsstLog


def countMaskedPixels(maskedIm, maskPlane):
    maskBit = maskedIm.mask.getPlaneBitMask(maskPlane)
    # bit = afwImage.Mask.getPlaneBitMask(maskPlane)
    # nPix = len(np.where(np.bitwise_and(maskedIm.mask.array, maskBit))[0])
    nPix = np.where(np.bitwise_and(maskedIm.mask.array, maskBit))[0].flatten().size
    return nPix


class PairedVisitListTaskRunner(pipeBase.TaskRunner):
    """Subclass of TaskRunner for handling intrinsically paired visits.

    This transforms the processed arguments generated by the ArgumentParser
    into the arguments expected by tasks which take visit pairs for their
    run() methods.

    Such tasks' run() methods tend to take two arguments,
    one of which is the dataRef (as usual), and the other is the list
    of visit-pairs, in the form of a list of tuples.
    This list is supplied on the command line as documented,
    and this class parses that, and passes the parsed version
    to the run() method.

    See pipeBase.TaskRunner for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Parse the visit list and pass through explicitly."""
        visitPairs = []
        for visitStringPair in parsedCmd.visitPairs:
            visitStrings = visitStringPair.split(",")
            if len(visitStrings) != 2:
                raise RuntimeError("Found {} visits in {} instead of 2".format(len(visitStrings),
                                                                               visitStringPair))
            try:
                visits = [int(visit) for visit in visitStrings]
            except Exception:
                raise RuntimeError("Could not parse {} as two integer visit numbers".format(visitStringPair))
            visitPairs.append(visits)

        return pipeBase.TaskRunner.getTargetList(parsedCmd, visitPairs=visitPairs, **kwargs)


def parseCmdlineNumberString(inputString):
    """Parse command line numerical expression sytax and return as list of int

    Take an input of the form "'1..5:2^123..126'" as a string, and return
    a list of ints as [1, 3, 5, 123, 124, 125, 126]
    """
    outList = []
    for subString in inputString.split("^"):
        mat = re.search(r"^(\d+)\.\.(\d+)(?::(\d+))?$", subString)
        if mat:
            v1 = int(mat.group(1))
            v2 = int(mat.group(2))
            v3 = mat.group(3)
            v3 = int(v3) if v3 else 1
            for v in range(v1, v2 + 1, v3):
                outList.append(int(v))
        else:
            outList.append(int(subString))
    return outList


class SingleVisitListTaskRunner(pipeBase.TaskRunner):
    """Subclass of TaskRunner for the MakeBrighterFatterKernelTask.

    This transforms the processed arguments generated by the ArgumentParser
    into the arguments expected by makeBrighterFatterKernelTask.run().

    makeBrighterFatterKernelTask.run() takes a two arguments,
    one of which is the dataRef (as usual), and the other is the list
    of visit-pairs, in the form of a list of tuples.
    This list is supplied on the command line as documented,
    and this class parses that, and passes the parsed version
    to the run() method.

    See pipeBase.TaskRunner for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Parse the visit list and pass through explicitly."""
        assert len(parsedCmd.visitList) == 1  # even if you give two on the cmdline you only get 1 here
        visits = parseCmdlineNumberString(parsedCmd.visitList[0])

        return pipeBase.TaskRunner.getTargetList(parsedCmd, visitList=visits, **kwargs)


class NonexistentDatasetTaskDataIdContainer(pipeBase.DataIdContainer):
    """A DataIdContainer for the tasks for which the output does
    not yet exist."""

    def makeDataRefList(self, namespace):
        """Compute refList based on idList.

        This method must be defined as the dataset does not exist before this
        task is run.

        Parameters
        ----------
        namespace
            Results of parsing the command-line.

        Notes
        -----
        Not called if ``add_id_argument`` called
        with ``doMakeDataRefList=False``.
        Note that this is almost a copy-and-paste of the vanilla implementation,
        but without checking if the datasets already exist,
        as this task exists to make them.
        """
        if self.datasetType is None:
            raise RuntimeError("Must call setDatasetType first")
        butler = namespace.butler
        for dataId in self.idList:
            refList = list(butler.subset(datasetType=self.datasetType, level=self.level, dataId=dataId))
            # exclude nonexistent data
            # this is a recursive test, e.g. for the sake of "raw" data
            if not refList:
                namespace.log.warn("No data found for dataId=%s", dataId)
                continue
            self.refList += refList


def checkExpLengthEqual(exp1, exp2, v1=None, v2=None, raiseWithMessage=False):
    """Check the exposure lengths of two exposures are equal.

    Parameters:
    -----------
    exp1 : `lsst.afw.image.exposure.ExposureF`
        First exposure to check
    exp2 : `lsst.afw.image.exposure.ExposureF`
        Second exposure to check
    v1 : `int` or `str`, optional
        First visit of the visit pair
    v2 : `int` or `str`, optional
        Second visit of the visit pair
    raiseWithMessage : `bool`
        If True, instead of returning a bool, raise a RuntimeError if exposure
    times are not equal, with a message about which visits mismatch if the
    information is available.

    Raises:
    -------
    RuntimeError
        Raised if the exposure lengths of the two exposures are not equal
    """
    expTime1 = exp1.getInfo().getVisitInfo().getExposureTime()
    expTime2 = exp2.getInfo().getVisitInfo().getExposureTime()
    if expTime1 != expTime2:
        if raiseWithMessage:
            msg = "Exposure lengths for visit pairs must be equal. " + \
                  "Found %s and %s" % (expTime1, expTime2)
            if v1 and v2:
                msg += " for visit pair %s, %s" % (v1, v2)
            raise RuntimeError(msg)
        else:
            return False
    return True


def validateIsrConfig(isrTask, mandatory=None, forbidden=None, desirable=None, undesirable=None,
                      checkTrim=True, logName=None):
    """Check that appropriate ISR settings have been selected for the task.

    Note that this checks that the task itself is configured correctly rather
    than checking a config.

    Parameters
    ----------
    isrTask : `lsst.ip.isr.IsrTask`
        The task whose config is to be validated

    mandatory : `iterable` of `str`
        isr steps that must be set to True. Raises if False or missing

    forbidden : `iterable` of `str`
        isr steps that must be set to False. Raises if True, warns if missing

    desirable : `iterable` of `str`
        isr steps that should probably be set to True. Warns is False, info if
    missing

    undesirable : `iterable` of `str`
        isr steps that should probably be set to False. Warns is True, info if
    missing

    checkTrim : `bool`
        Check to ensure the isrTask's assembly subtask is trimming the images.
    This is a separate config as it is very ugly to do this within the
    normal configuration lists as it is an option of a sub task.

    Raises
    ------
    RuntimeError
        Raised if ``mandatory`` config parameters are False,
        or if ``forbidden`` parameters are True.

    TypeError
        Raised if parameter ``isrTask`` is an invalid type.

    Notes
    -----
    Logs warnings using an isrValidation logger for desirable/undesirable
    options that are of the wrong polarity or if keys are missing.
    """
    if not isinstance(isrTask, ipIsr.IsrTask):
        raise TypeError(f'Must supply an instance of ipIsr.IsrTask not {type(isrTask)}')

    configDict = isrTask.config.toDict()

    if logName and isinstance(logName, str):
        log = lsstLog.getLogger(logName)
    else:
        log = lsstLog.getLogger("isrValidation")

    if mandatory:
        for configParam in mandatory:
            if configParam not in configDict:
                raise RuntimeError(f"Mandatory parameter {configParam} not found in the isr configuration.")
            if configDict[configParam] is False:
                raise RuntimeError(f"Must set config.isr.{configParam} to True for this task.")

    if forbidden:
        for configParam in forbidden:
            if configParam not in configDict:
                log.warn(f"Failed to find key {configParam} in the isr config. This is set to be " +
                         "forbidden, so you should ensure that the equivalent for your obs_package to False.")
                continue
            if configDict[configParam] is True:
                raise RuntimeError(f"Must set config.isr.{configParam} to False for this task.")

    if desirable:
        for configParam in desirable:
            if configParam not in configDict:
                log.info(f"Failed to find key {configParam} in the isr config. You probably want" +
                         " to set the equivalent for your obs_package to True.")
                continue
            if configDict[configParam] is False:
                log.warn(f"Found config.isr.{configParam} set to False for this task." +
                         " It is probably desirable to have this set to True")
    if undesirable:
        for configParam in undesirable:
            if configParam not in configDict:
                log.info(f"Failed to find key {configParam} in the isr config. You probably want" +
                         " to set the equivalent for your obs_package to False.")
                continue
            if configDict[configParam] is True:
                log.warn(f"Found config.isr.{configParam} set to True for this task." +
                         " It is probably desirable to have this set to False")

    if checkTrim:  # subtask setting, seems non-trivial to combine with above lists
        if not isrTask.assembleCcd.config.doTrim:
            raise RuntimeError("Must trim when assembling CCDs. Set config.isr.assembleCcd.doTrim to True")
