    def _xcorr(self, im1, im2, gains):
        """Calculate the cross-correlation of two images im1 and im2 using robust measures of the covariance.

        Maximum lag is maxLag, and ignore border pixels around the outside.
        Sigma is the number of sigma passed to sig cut.
        GAIN allows user specified GAINS to be used otherwise the default gains are used.
        The biasCorr parameter is used to correct from the bias of our measurements introduced by the sigma cuts.
        This was calculated using the sim. code at the bottom.
        This function returns one quater of the correlation function, the sum of the means of the two images and
        the individual means of the images
        """
        maxLag = self.config.maxLag
        border = self.config.nPixBorderXCorr
        sigma = self.config.nSigmaClipXCorr
        biasCorr = self.config.nPixBorderXCorr

        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(sigma)

        means = [None, None]
        means1 = [None, None]
        for imNum, im in enumerate([im1, im2]):
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
            means[imNum] = afwMath.makeStatistics(im[border:-border, border:-border],
                                                  afwMath.MEANCLIP, sctrl).getValue()
            temp = im.clone()
            # Rescale each amp by the appropriate gain and subtract the mean.
            for ampNum, amp in enumerate(ccd):
                # smi = im[amp.getDataSec(True)]
                # smiTemp = temp[amp.getDataSec(True)]
                smi = im[amp.getBBox()]
                smiTemp = temp[amp.getBBox()]
                mean = afwMath.makeStatistics(smi, afwMath.MEANCLIP, sctrl).getValue()
                gain = gains[ampNum]
                # gain/=gain
                smi *= gain
                print(mean*gain, afwMath.makeStatistics(smi, afwMath.MEANCLIP, sctrl).getValue())
                smi -= mean*gain
                smiTemp *= gain
            means1[imNum] = afwMath.makeStatistics(temp[border:-border, border:-border],
                                                   afwMath.MEANCLIP, sctrl).getValue()
            print(afwMath.makeStatistics(temp[border:-border, border:-border],
                                         afwMath.MEANCLIP, sctrl).getValue())
        #    print(afwMath.makeStatistics(temp, afwMath.MEANCLIP,sctrl).getValue()-
        #          afwMath.makeStatistics(temp[0:-maxLag,0:-maxLag], afwMath.MEANCLIP,sctrl).getValue())

        #
        # Actually diff the images
        #
        diff = im1.clone()
        diff = diff.getMaskedImage().getImage()
        diff -= im2.getMaskedImage().getImage()

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
        dim0 = diff[0: -maxLag, : -maxLag]
        dim0 -= afwMath.makeStatistics(dim0, afwMath.MEANCLIP, sctrl).getValue()
        w, h = dim0.getDimensions()
        xcorr = afwImage.ImageF(maxLag + 1, maxLag + 1)
        for xlag in range(maxLag + 1):
            for ylag in range(maxLag + 1):
                dim_xy = diff[xlag:xlag + w, ylag: ylag + h].clone()
                dim_xy -= afwMath.makeStatistics(dim_xy, afwMath.MEANCLIP, sctrl).getValue()
                dim_xy *= dim0
                xcorr[xlag, ylag] = afwMath.makeStatistics(dim_xy, afwMath.MEANCLIP, sctrl).getValue()/(biasCorr)

        # L = np.shape(xcorr.getArray())[0]-1
        # XCORR = np.zeros([2*L+1, 2*L+1])
        xcorr_full = self._tileArray(xcorr)
        # for i in range(L+1):
        #     for j in range(L+1):
        #         XCORR[i+L, j+L] = xcorr.getArray()[i, j]
        #         XCORR[-i+L, j+L] = xcorr.getArray()[i, j]
        #         XCORR[i+L, -j+L] = xcorr.getArray()[i, j]
        #         XCORR[-i+L, -j+L] = xcorr.getArray()[i, j]
        print(sum(means1), xcorr.getArray()[0, 0], np.sum(xcorr_full), xcorr.getArray()[0, 0]/sum(means1),
              np.sum(xcorr_full)/sum(means1))
        return (xcorr, means1)
