import eups

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats
import os, sys, time, datetime, glob
from subprocess import *
import pickle as pkl

from lsst.daf.persistence import Butler
from lsst.cp.pipe.makeBrighterFatterKernel import MakeBrighterFatterKernelTask
import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.daf.persistence import Butler
from lsst.ip.isr.isrTask import IsrTask, IsrTaskConfig
from lsst.ip.isr.isrFunctions import brighterFatterCorrection
from lsst.meas.algorithms import SourceDetectionTask
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig
from lsst.geom import Point2I, Box2I

spots_already_done = True
flats_already_done = True
new_spot_repo = False

# Set up things at the beginning
SPOTS_REPO_DIR = "spots_repo_bip_base"
SPOTS_DIR = "/mnt/storm-lsst/GUI/20181207_e2v_bf-30um/"
spots_sequence_numbers = list(np.arange(200,230))
FLATS_REPO_DIR = "flats_repo_bip_base"
FLATS_DIR = "/mnt/storm-lsst/GUI/20181205_e2v_flats/"
flats_sequence_numbers = list(np.arange(100,300,10))+list(np.arange(101,301,10))
RAFT = "R00" # "R00 for E2V, R02 for ITL 002
DETECTOR = 0 # 0 for E2V, 2 for ITL

# These are for doing the spot size cuts on the catalog
minSize = 0.8
maxSize = 2.5

# These are for the BF slope plotting
minSpot = 1.4
maxSpot = 1.7

plotCorrection = True

# This index ignores results beyond which the spots are saturated
max_flux_index = 30
# These indices determine which values are used for the slope calculation
min_slope_index = 5
max_slope_index = 25

# Also set up the region we want to characterize.  This is chosen to avoid bad segments and keep away from
# optical distortions at the edge.  There are ~ 10,000 spots in this region
LL = geom.Point2I(2050,500); UR = geom.Point2I(2550,3500);
spots_bbox = geom.Box2I(LL,UR)

# Calculate errors assuming some portion is systematic and some portion in statistical.
# Statistical errors divide down by sqrt(N), systematic errors don't.
# More work is needed to better quantify the systematic fraction
syst_fraction = 0.10

# First ingest the flats images
if not flats_already_done:
    step1 = Popen("rm -rf %s"%FLATS_REPO_DIR, shell=True)
    Popen.wait(step1)
    step2 = Popen("mkdir -p %s"%FLATS_REPO_DIR, shell=True)
    Popen.wait(step2)
    step3 = Popen('echo "lsst.obs.lsst.ucd.UcdMapper" > %s/_mapper'%FLATS_REPO_DIR, shell=True)
    Popen.wait(step3)

    fitsFileList = []
    for seqnum in flats_sequence_numbers:
        fitsFileList += glob.glob(FLATS_DIR+'*_flat_flat_%d_201???????????.fits'%seqnum)
    args = FLATS_REPO_DIR
    for fitsFile in fitsFileList:
        args += " "
        args += fitsFile
    # Ingest the target images
    ingest = Popen('ingestImages.py '+ args, shell=True)
    Popen.wait(ingest)
    print("finished ingesting flats images")

    # Now calculate the BF kernel

    # First get the gains calculated correctly
    #GAIN_REPO_DIR = "flats_repo_itl/rerun/test"
    #gain_butler = Butler(GAIN_REPO_DIR)
    #gain_data = gain_butler.get('brighterFatterGain', dataId={'raftName': RAFT, 'detectorName': 'S00', 'detector': DETECTOR})    
    
    flats_butler = Butler(FLATS_REPO_DIR)
    #flats_butler.put(gain_data, 'brighterFatterGain')

    visits = []
    my_metaData = flats_butler.queryMetadata('raw', ['visit', 'dateObs'])
    for item in my_metaData:
        visits.append(item[0])
    pairs = []
    for i in range(0,len(visits),2):
        pairs.append('%s,%s'%(str(visits[i]),str(visits[i+1])))
    args = [FLATS_REPO_DIR, '--rerun', 'test','--id', 'detector=%d'%DETECTOR,'--visit-pairs']
    for pair in pairs:
        args.append(str(pair))

    args = args + ['-c','xcorrCheckRejectLevel=2', 'doCalcGains=True', 'level="AMP"',
                   '--clobber-config', '--clobber-versions']
    command_line = 'makeBrighterFatterKernel.py ' + ' '.join(args)
    print(command_line)
    corr_struct = MakeBrighterFatterKernelTask.parseAndRun(args=args)
flats_butler = Butler(FLATS_REPO_DIR+'/rerun/test')
bf_kernel = flats_butler.get('brighterFatterKernel', dataId={'raftName': RAFT, 'detectorName': 'S00', 'detector': DETECTOR})
gain_data = flats_butler.get('brighterFatterGain', dataId={'raftName': RAFT, 'detectorName': 'S00', 'detector': DETECTOR})

# Now we shift to the spots data
# These setup the image characterization and ISR
isrConfig = IsrTaskConfig()
isrConfig.doBias = False
isrConfig.doDark = False
isrConfig.doFlat = False
isrConfig.doFringe = False
isrConfig.doDefect = False
isrConfig.doAddDistortionModel = False
isrConfig.doWrite = True
isrConfig.doAssembleCcd = True
isrConfig.expectWcs = False
isrConfig.doLinearize = False

charConfig = CharacterizeImageConfig()
charConfig.installSimplePsf.fwhm = 0.05
charConfig.doMeasurePsf = False
charConfig.doApCorr = False
charConfig.doDeblend = False
charConfig.repair.doCosmicRay = False
charConfig.repair.doInterpolate = False      
charConfig.detection.background.binSize = 128
charConfig.detection.minPixels = 5

# Now we characterize the spot sizes
if not spots_already_done:
    if new_spot_repo:
        step1 = Popen("rm -rf %s"%SPOTS_REPO_DIR, shell=True)
        Popen.wait(step1)
        step2 = Popen("mkdir -p %s/rerun/test/plots"%SPOTS_REPO_DIR, shell=True)
        Popen.wait(step2)
        step3 = Popen('echo "lsst.obs.lsst.ucd.UcdMapper" > %s/_mapper'%SPOTS_REPO_DIR, shell=True)
        Popen.wait(step3)

    # Then ingest the spots images
    fitsFileList = []
    for seqnum in spots_sequence_numbers:
        fitsFileList += glob.glob(SPOTS_DIR+'*_spot_spot_%d_201???????????.fits'%seqnum)
    args = SPOTS_REPO_DIR
    for fitsFile in fitsFileList:
        args += " "
        args += fitsFile
    # Ingest the target images
    ingest = Popen('ingestImages.py '+ args, shell=True)
    Popen.wait(ingest)
    print("finished ingesting spots images")
    spots_butler = Butler(SPOTS_REPO_DIR)

    visits = []
    my_metaData = spots_butler.queryMetadata('raw', ['visit', 'dateObs'])
    for item in my_metaData:
        visits.append(item[0])
    byamp_results = []
    byamp_corrected_results = []
    for visit in visits:
        print("Getting exposure # %d"%visit)
        sys.stdout.flush()
        exposure = spots_butler.get('raw', dataId={'visit': visit, 'detector': DETECTOR})
        # Perform the instrument signature removal (mainly assembling the CCD)
        isrTask = IsrTask(config=isrConfig)
        exposure_isr = isrTask.run(exposure).exposure
        # For now, we're applying the gain manually
        ccd = exposure_isr.getDetector()
        for do_bf_corr in [False, True]:
            exposure_copy=exposure_isr.clone()            
            for amp in ccd:
                if spots_bbox.overlaps(amp.getBBox()):
                    gain = gain_data[amp.getName()]
                    img = exposure_copy.image
                    sim = img.Factory(img, amp.getBBox())
                    sim *= gain
                    print(amp.getName(), gain, amp.getBBox())
                    sys.stdout.flush()                    
                    if do_bf_corr:
                        brighterFatterCorrection(exposure_copy[amp.getBBox()],bf_kernel.kernel[amp.getName()],20,10,False)
                else:
                    continue
            # Now trim the exposure down to the region of interest
            trimmed_exposure = exposure_copy[spots_bbox]

            # Now find and characterize the spots
            charTask = CharacterizeImageTask(config=charConfig)
            tstart=time.time()
            charResult = charTask.run(trimmed_exposure)
            spotCatalog = charResult.sourceCat
            print("%s, Correction = %r, Characterization took "%(amp.getName(),do_bf_corr),str(time.time()-tstart)[:4]," seconds")
            sys.stdout.flush()                                
            # Now trim out spots not between minSize and maxSize
            select = ((spotCatalog['base_SdssShape_xx'] >= minSize) & (spotCatalog['base_SdssShape_xx'] <= maxSize) & 
              (spotCatalog['base_SdssShape_yy'] >= minSize) & (spotCatalog['base_SdssShape_yy'] <= maxSize))
            spotCatalog  = spotCatalog.subset(select)
            x2 = spotCatalog['base_SdssShape_xx']
            y2 = spotCatalog['base_SdssShape_yy']
            flux = spotCatalog['base_SdssShape_instFlux']
            numspots = len(flux)
            print("Detected ",len(spotCatalog)," objects, Flux = %f, X2 = %f, Y2 = %f"%(np.nanmean(flux),np.nanmean(x2),np.nanmean(y2)))
            sys.stdout.flush()                                
            if do_bf_corr:
                byamp_corrected_results.append([numspots, np.nanmean(flux), np.nanstd(flux), np.nanmean(x2), np.nanstd(x2),
                                   np.nanmean(y2), np.nanstd(y2)])
            else:
                byamp_results.append([numspots, np.nanmean(flux), np.nanstd(flux), np.nanmean(x2), np.nanstd(x2),
                                   np.nanmean(y2), np.nanstd(y2)])
    spots_pickle = {'results':byamp_results, 'corrected_results': byamp_corrected_results}
    filename = SPOTS_REPO_DIR+"/rerun/test/spots_results.pkl"
    with open(filename, 'wb') as f:
        pkl.dump(spots_pickle, f)

# Now plot the result
# A little bit of a fudge here in that the slope in percent per 50K electrons
# has been based on the peak flux, and the DM stack returns the total flux.
# I have applied an estimated correction factor of 6.4 to compensate, but this is only
# approximate.  However, the relative slopes corrected and uncorrected are not impacted by this.
spots_butler = Butler(SPOTS_REPO_DIR)
visits = []
my_metaData = spots_butler.queryMetadata('raw', ['visit', 'dateObs'])
for item in my_metaData:
    visits.append(item[0])

exposure = spots_butler.get('raw', dataId={'visit': visits[0], 'detector': DETECTOR})
ccd = exposure.getDetector()

filename = SPOTS_REPO_DIR+"/rerun/test/spots_results.pkl"
with open(filename, 'rb') as f:
    spots_pickle= pkl.load(f)


textDelta = (maxSpot - minSpot) / 20
# These next are in case not all fluxes produced good results
byamp_results = spots_pickle['results']
try:
    results = np.array([byamp_results[i] for i in range(max_flux_index)])
    max_slope_ind = max_slope_index
except:
    results = np.array(byamp_results)
    max_slope_ind = min(len(results) - 4, max_slope_index)
xerror = results[:,2]/np.sqrt(results[:,0])
xyerror = results[:,4] * (syst_fraction + (1 - syst_fraction) / np.sqrt(results[:,0]))
yyerror = results[:,6] * (syst_fraction + (1 - syst_fraction) / np.sqrt(results[:,0]))

if plotCorrection:
    byamp_corrected_results = spots_pickle['corrected_results']
    try:
        corrected_results = np.array([byamp_corrected_results[i] for i in range(max_flux_index)])
        max_slope_ind_corr = max_slope_index
    except:
        corrected_results = np.array(byamp_corrected_results)           
        max_slope_ind_corr = min(len(corrected_results) - 4, max_slope_index)

    corrected_xerror = corrected_results[:,2]/np.sqrt(corrected_results[:,0])
    corrected_xyerror = corrected_results[:,4] * (syst_fraction + (1 - syst_fraction) / np.sqrt(corrected_results[:,0]))
    corrected_yyerror = corrected_results[:,6] * (syst_fraction + (1 - syst_fraction) / np.sqrt(corrected_results[:,0]))

plt.figure(figsize=(16,8))
plt.title("Brighter-Fatter - 30 micron Spots", fontsize = 36)
# First plot the uncorrected data
plt.errorbar(results[:,1], results[:,3], xerr = xerror, 
             yerr = xyerror, color = 'green', lw = 2, label = 'X2', ls='', marker='x')
plt.errorbar(results[:,1], results[:,5], xerr = xerror, 
             yerr = yyerror, color = 'red', lw = 2, label = 'Y2', ls='',marker='x')
slope, intercept, r_value, p_value, std_err = stats.linregress(results[min_slope_index:max_slope_ind,1], results[min_slope_index:max_slope_ind,3])
xplot=np.linspace(-5000.0,3200000.0,100)
yplot = slope * xplot + intercept
plt.plot(xplot, yplot, color='green', lw = 2, ls = '--')
tslope = slope * 100.0 * 200000.0
plt.text(10000.0,maxSpot-textDelta,"X Slope = %.2f %% per 50K e-"%tslope, fontsize=24)

slope, intercept, r_value, p_value, std_err = stats.linregress(results[min_slope_index:max_slope_ind,1], results[min_slope_index:max_slope_ind,5])
xplot=np.linspace(-5000.0,3200000.0,100)
yplot = slope * xplot + intercept
plt.plot(xplot, yplot, color='red', lw = 2, ls = '--')
tslope = slope * 100.0 * 200000.0
plt.text(10000.0,maxSpot-2*textDelta,"Y Slope = %.2f %% per 50K e-"%tslope, fontsize=24)

if plotCorrection:
    # Now plot the corrected data
    plt.errorbar(corrected_results[:,1], corrected_results[:,3], xerr = corrected_xerror, 
                yerr = corrected_xyerror, color = 'cyan', lw = 2, label = 'Corrected X2')
    plt.errorbar(corrected_results[:,1], corrected_results[:,5], xerr = corrected_xerror,
                yerr = corrected_yyerror, color = 'magenta', lw = 2, label = 'Corrected Y2')
    slope, intercept, r_value, p_value, std_err = stats.linregress(corrected_results[min_slope_index:max_slope_ind_corr,1], corrected_results[min_slope_index:max_slope_ind_corr,3])
    xplot=np.linspace(-5000.0,3200000.0,100)
    yplot = slope * xplot + intercept
    plt.plot(xplot, yplot, color='cyan', lw = 2, ls = '--')
    tslope = slope * 100.0 * 200000.0
    plt.text(10000.0,maxSpot-3*textDelta,"Corrected X Slope = %.2f %% per 50K e-"%tslope, fontsize=24)

    slope, intercept, r_value, p_value, std_err = stats.linregress(corrected_results[min_slope_index:max_slope_ind_corr,1], corrected_results[min_slope_index:max_slope_ind_corr,5])
    xplot=np.linspace(-5000.0,3200000.0,100)
    yplot = slope * xplot + intercept
    plt.plot(xplot, yplot, color='magenta', lw = 2, ls = '--')
    tslope = slope * 100.0 * 200000.0
    plt.text(10000.0,maxSpot-4*textDelta,"Corrected Y Slope = %.2f %% per 50K e-"%tslope, fontsize=24)

plt.xlim(0.0,1000000.0)
plt.xticks([0,500000,1000000])
plt.ylim(minSpot, maxSpot)
plt.xlabel('Spot Flux(electrons)',fontsize=24)
plt.ylabel('Second Moment (Pixels)',fontsize=24)
plt.legend(loc= 'lower right',fontsize = 18)
plt.savefig(SPOTS_REPO_DIR+"/rerun/test/plots/BF_Slopes_Corrected.pdf")
plt.close('all')

