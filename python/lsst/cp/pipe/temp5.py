    def xcorr(self, im1, im2, Visits, n=5, border=20, frame=None, CCD=[1], GAIN=None, sigma=5, biasCorr=0.9241):
        """Calculate the cross-correlation of two images im1 and im2 (using robust measures of the covariance).

        This is designed to be called through xcorrFromVisit as that performs some simple ISR.
        Maximum lag is n, and ignore border pixels around the outside. Sigma is the number of sigma passed
        to sig cut.
        GAIN allows user specified GAINS to be used otherwise the default gains are used.
        The biasCorr parameter is used to correct from the bias of our measurements introduced by the sigma cuts.
        This was calculated using the sim. code at the bottom.
        This function returns one quater of the correlation function, the sum of the means of the two images and
        the individual means of the images
        """
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(sigma)
        ims = [im1, im2]
        means = [None, None]
        means1 = [None, None]
        for i, im in enumerate(ims):
            # ccd = afwCG.cast_Ccd(im.getDetector())
            ccd = im.getDetector()
            try:
                frameId = int(re.sub(r"^SUPA0*", "", im.getMetadata().get("FRAMEID")))
            except:
                frameId = -1
            #
            # Starting with an Exposure, MaskedImage, or Image trim the data and convert to float
            #
            for attr in ("getMaskedImage", "getImage"):
                if hasattr(im, attr):
                    im = getattr(im, attr)()
            try:
                im = im.convertF()
            except AttributeError:
                pass
            # im = trim(im, ccd)
            means[i] = afwMath.makeStatistics(im[border:-border, border:-border],
                                              afwMath.MEANCLIP, sctrl).getValue()
            temp = im.clone()
            # Rescale each amp by the appropriate gain and subtract the mean.
            for j, a in enumerate(ccd):
                # smi = im[a.getDataSec(True)]
                # smiTemp = temp[a.getDataSec(True)]
                smi = im[a.getBBox()]
                smiTemp = temp[a.getBBox()]
                mean = afwMath.makeStatistics(smi, afwMath.MEANCLIP, sctrl).getValue()
                if GAIN == None:
                    # gain = a.getElectronicParams().getGain()
                    gain = a.getGain()
                else:
                    gain = GAIN[j]
                # gain/=gain
                smi *= gain
                print(mean*gain, afwMath.makeStatistics(smi, afwMath.MEANCLIP, sctrl).getValue())
                smi -= mean*gain
                smiTemp *= gain
            means1[i] = afwMath.makeStatistics(
                temp[border:-border, border:-border], afwMath.MEANCLIP, sctrl).getValue()
            print(afwMath.makeStatistics(temp[border:-border, border:-border],
                                         afwMath.MEANCLIP, sctrl).getValue())
        #    print(afwMath.makeStatistics(temp, afwMath.MEANCLIP,sctrl).getValue()-
        #          afwMath.makeStatistics(temp[0:-n,0:-n], afwMath.MEANCLIP,sctrl).getValue())
        im1, im2 = ims
        #
        # Actually diff the images
        #
        diff = ims[0].clone()
        diff = diff.getMaskedImage().getImage()
        diff -= ims[1].getMaskedImage().getImage()

        diff = diff[border:-border, border:-border]
        # diff.writeFits("./Data/Diff_CCD_"+str(CCD)+".fits")
        #
        # Subtract background.  It should be a constant, but it isn't always
        #
        binsize = 128
        nx = diff.getWidth()//binsize
        ny = diff.getHeight()//binsize
        bctrl = afwMath.BackgroundControl(nx, ny, sctrl, afwMath.MEANCLIP)
        bkgd = afwMath.makeBackground(diff, bctrl)
        diff -= bkgd.getImageF(afwMath.Interpolate.CUBIC_SPLINE, afwMath.REDUCE_INTERP_ORDER)
        # diff.writeFits("./Data/Diff_backsub_CCD_"+str(CCD)+".fits")
        if frame is not None:
            ds9.mtv(diff, frame=frame, title="diff")

        if False:
            global diffim
            diffim = diff
        if False:
            print(afwMath.makeStatistics(diff, afwMath.MEDIAN, sctrl).getValue())
            print(afwMath.makeStatistics(diff, afwMath.VARIANCECLIP, sctrl).getValue(), np.var(diff.getArray()))
        #
        # Measure the correlations
        #
        dim0 = diff[0: -n, : -n]
        dim0 -= afwMath.makeStatistics(dim0, afwMath.MEANCLIP, sctrl).getValue()
        w, h = dim0.getDimensions()
        xcorr = afwImage.ImageF(n + 1, n + 1)
        for di in range(n + 1):
            for dj in range(n + 1):
                dim_ij = diff[di:di + w, dj: dj + h].clone()
                dim_ij -= afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()
                dim_ij *= dim0
                xcorr[di, dj] = afwMath.makeStatistics(dim_ij, afwMath.MEANCLIP, sctrl).getValue()/(biasCorr)
        L = np.shape(xcorr.getArray())[0]-1
        XCORR = np.zeros([2*L+1, 2*L+1])
        for i in range(L+1):
            for j in range(L+1):
                XCORR[i+L, j+L] = xcorr.getArray()[i, j]
                XCORR[-i+L, j+L] = xcorr.getArray()[i, j]
                XCORR[i+L, -j+L] = xcorr.getArray()[i, j]
                XCORR[-i+L, -j+L] = xcorr.getArray()[i, j]
        print(sum(means1), xcorr.getArray()[0, 0], np.sum(XCORR), xcorr.getArray()[0, 0]/sum(means1),
              np.sum(XCORR)/sum(means1))
        return (xcorr, means1)
