import lsst.afw.math as afwMath
import numpy as np

def xcorrFromVisitPair(self, dataRef, v1, v2):
    """Return the cross-correlation from a given pair of visits.

    This is code preforms some preliminary operations and then calls the main correlation calc code.
    This is used for calculating the xcorr after setting the gains.

    Parameters:
    -----------
    butler : `lsst.daf.persistence.butler`
        Butler for the repo containg the eotest data to be used

    Returns:
    xcorrImg : `np.array`

    means : `list` of `float`
        The sigma-clipped-mean flux in the input images

    """
    ims = [self.isr(dataRef, v1), self.isr(dataRef, v2)]

    xcorrImg, xcorrMeans = self._xcorr(im1, im2, gains)

    # TODO: Change to lsstDebug
    if False:
        means = [afwMath.makeStatistics(im, afwMath.MEANCLIP).getValue() for im in ims]
        self._plotXcorr(xcorrImg.clone(), (xcorrMeans[0]+means[1]),
                        title=r"Visits %s; %s, CCDs %s  $\langle{I}\rangle"
                        " = %.3f$ (%s) Var = %.4f" %
                        (self._getNameOfSet(v1), self._getNameOfSet(v2), self._getNameOfSet(ccds),
                         (xcorrMeans[0]+means[1]), ims[0].getFilter().getName(), float(xcorrImg.getArray()[0, 0]) /
                         (xcorrMeans[0]+means[1])), zmax=zmax, fig=fig, SAVE=True,
                        fileName=(os.path.join(OUTPUT_PATH, ("Xcorr_visit_" + str(v1[0])+"_"+str(v2[0])+"_ccd_" +
                                                             str(ccds[0])+".png"))))
    return xcorrImg, xcorrMeans
