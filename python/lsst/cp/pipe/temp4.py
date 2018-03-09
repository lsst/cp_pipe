import lsst.afw.math as afwMath
import numpy as np

def xcorrFromVisitPair(self, butler, v1, v2, 
                       xxx_remove_ccds=[1],
                       xxx_rename_n=5,
                       xxx_check_for_override_border=10,
                       xxx_pexConfig_plot=False,
                       zmax=.04, fig=None, display=False, GAIN=None, sigma=5):
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
    for i, vs in enumerate([v1, v2, ]):
        for v in vs:
            for ccd in ccds:
                tmp = isr(butler, v, ccd)
                if ims[i] is None:
                    ims[i] = tmp
                    im = ims[i].getMaskedImage()
                else:
                    im += tmp.getMaskedImage()

        nData = len(ccds)*len(vs)
        if nData > 1:
            im /= nData
        means[i] = afwMath.makeStatistics(im, afwMath.MEANCLIP).getValue()
        # if display:  # TODO: Move to lsstDebug, use proper display
            # ds9.mtv(trim(ims[i]), frame=i, title=v)

    ims = [self.isr(butler, v1, ccd), self.isr(butler, v2, ccd)]
    means = [afwMath.makeStatistics(im, afwMath.MEANCLIP).getValue() for im in ims]

    # mean = np.mean(means)
    xcorrImg, means1 = self._xcorr(*ims, Visits=[v1, v2], n=n, border=border, frame=len(ims)
                                   if display else None, CCD=[ccds[0]], GAIN=GAIN, sigma=sigma)

    # TODO: Change to lsstDebug
    if plot:
        self._plotXcorr(xcorrImg.clone(), (means1[0]+means[1]), title=r"Visits %s; %s, CCDs %s  $\langle{I}\rangle"
                        " = %.3f$ (%s) Var = %.4f" %
                        (self._getNameOfSet(v1), _getNameOfSet(v2), _getNameOfSet(ccds),
                         (means1[0]+means[1]), ims[0].getFilter().getName(), float(xcorrImg.getArray()[0, 0]) /
                         (means1[0]+means[1])), zmax=zmax, fig=fig, SAVE=True,
                        fileName=(os.path.join(OUTPUT_PATH, ("Xcorr_visit_" + str(v1[0])+"_"+str(v2[0])+"_ccd_" +
                                                             str(ccds[0])+".png"))))
    return xcorrImg, means1
