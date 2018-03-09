import numpy as np

    def generateKernel(self, corrs, means):
        """Generate the full kernel from a list of (gain-corrected) cross-correlations and means.

        Taking a list of quarter-image, gain-corrected cross-correlations, do a pixel-wise sigma-clipped
        mean of each, and tile into the full-sized kernel image.

        Each corr in corrs is one quarter of the full cross-correlation, and has been gain-corrected.
        Each mean in means is a tuple of the means of the two individual images, corresponding to that corr.

        config.xCorrSumRejectLevel essentially is a sanity check parameter.
        If this condition is violated there is something unexpected going on in the image, and it is
        discarded from the stack before the clipped-mean is calculated.

        Parameters:
        -----------
        corrs : `list` of `np.array`
            A list of the quarter-image cross-correlations
        means : `list` of `tuples` of `floats`
            The means of the input images for each corr in corrs

        Returns:
        --------
        kernel : `np.array`
            The output kernel
        """
        if not isinstance(corrs, list):  # we expect a list of arrays
            corrs = [corrs]

        # Try to average over a set of possible inputs. This generates a simple function of the kernel that
        # should be constant across the images, and averages that.
        xcorrList = []
        sctrl = afwMath.StatisticsControl()
        sctrl.setNumSigmaClip(self.config.kernelGenSigmaClip)

        for corrNum, ((mean1, mean2), corr) in enumerate(zip(means, corrs)):
            corr[0, 0] -= (mean1+mean2)
            if corr[0, 0] > 0:
                self.log.warn('Skipped item %s due to unexpected value of (variance-mean)!'%corrNum)
                continue
            corr /= -float(1.0*(mean1**2+mean2**2))

            fullCorr = self._tileArray(corr)

            xcorrSum = np.abs(np.sum(fullCorr))/np.sum(np.abs(fullCorr))  # xx why call abs() twice?
            if xcorrSum > config.xCorrSumRejectLevel:
                self.log.warn("Sum of the xcorr is unexpectedly high. Investigate item num %s. \n"
                              "value = %s"%(corrNum, xcorrSum))
                continue
            xcorrList.append(fullCorr)

        # stack the individual xcorrs and apply a per-pixel clipped-mean
        meanXcorr = np.zeros_like(fullCorr)
        xcorrList = np.transpose(xcorrList)
        for i in range(np.shape(meanXcorr)[0]):
            for j in range(np.shape(meanXcorr)[1]):
                meanXcorr[i, j] = afwMath.makeStatistics(xcorrList[i, j], afwMath.MEANCLIP, sctrl).getValue()

        return self._SOR(meanXcorr)

    def _SOR(self, source, dx=1.0):
        """An implementation of the successive over relaxation (SOR) method.

        self.config.maxIterSOR and self.config.eLevelSOR are parameters for deciding when to end the SOR,
        either after a certain number of iterations, or after the error has been
        reduced by a factor self.config.eLevelSOR
        """
        # initialise function: Done to zero here. Setting boundary conditions too!
        # xxx is this the right dtype for these? Maybe specify float64/128 as these are very small.
        func = np.zeros([source.shape[0]+2, source.shape[1]+2])
        resid = np.zeros([source.shape[0]+2, source.shape[1]+2])
        rhoSpe = np.cos(np.pi/source.shape[0])  # Here a square grid is assummed

        inError = 0
        # Calculate the initial error
        for i in range(1, func.shape[0]-1):
            for j in range(1, func.shape[1]-1):
                resid[i, j] = (func[i, j-1]+func[i, j+1]+func[i-1, j] +
                               func[i+1, j]-4*func[i, j]-source[i-1, j-1])
        inError = np.sum(np.abs(resid))
        counter = 0
        omega = 1.0

        # Iterate until convergence
        # We perform two sweeps per cycle, updating 'odd' and 'even' points separately
        while counter < self.config.maxIterSOR*2:
            outError = 0
            if counter%2 == 0:
                for i in range(1, func.shape[0]-1, 2):
                    for j in range(1, func.shape[0]-1, 2):
                        resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                            func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0]-1, 2):
                    for j in range(2, func.shape[0]-1, 2):
                        resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                            func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                        func[i, j] += omega*resid[i, j]*.25
            else:
                for i in range(1, func.shape[0]-1, 2):
                    for j in range(2, func.shape[0]-1, 2):
                        resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                            func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                        func[i, j] += omega*resid[i, j]*.25
                for i in range(2, func.shape[0]-1, 2):
                    for j in range(1, func.shape[0]-1, 2):
                        resid[i, j] = float(func[i, j-1]+func[i, j+1]+func[i-1, j] +
                                            func[i+1, j]-4.0*func[i, j]-dx*dx*source[i-1, j-1])
                        func[i, j] += omega*resid[i, j]*.25
            outError = np.sum(np.abs(resid))
            if outError < inError*self.config.eLevelSOR:
                break
            if counter == 0:
                omega = 1.0/(1-rhoSpe*rhoSpe/2.0)
            else:
                omega = 1.0/(1-rhoSpe*rhoSpe*omega/4.0)
            counter += 1

        if counter >= self.config.maxIterSOR*2:
            self.log.warn("Did not converge in %s iterations.\noutError: %s, inError: "
                          "%s,"%(counter//2, outError, inError*self.config.eLevelSOR))
        else:
            self.log.warn("Converged in %s iterations.\noutError: %s, inError: "
                          "%s", counter//2, outError, inError*self.config.eLevelSOR)
        return func[1:-1, 1:-1]
