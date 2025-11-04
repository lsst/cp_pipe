BPS Template Files for Calibration Construction
===============================================

This directory contains BPS template files for calibration testing and
construction. These are meant to conform to the naming conventions laid out in
[DMTN-222](https://dmtn-222.lsst.io).

Overview
--------

These template files are only part of calibration construction. They are meant
to be flexible enough to use for both test construction into a user collection,
as well as final construction into the overall collection.

The templates rely on a lot of exported environment variables, defined
below. Note that because the bps file does not share an env with the caller all
the env vars must be explicitly exported.

The calibrations also assume that the caller is continuously certifying
calibrations for the next step into a collection
`${USER_CALIB_PREFIX}${INSTRUMENT}/calib/${TICKET}/${TAG}`.

More information on bps is [here](https://pipelines.lsst.io/modules/lsst.ctrl.bps/quickstart.html).

Environment Variables
---------------------

These are the environment variables that must be used. Examples are substituted in.

* `export USER_CALIB_PREFIX=""` or `export USER_CALIB_PREFIX=u/erykoff/`: Set to a empty string for official calibrations, or the user collection prefix (**with trailing slash**).
* `export INSTRUMENT=LSSTComCam`: The name of the instrument.
* `export TICKET=DM-46562`: The name of the ticket assocated with the calib construction.
* `export REPO=/repo/main`: The name of the butler repository to generate calibs.
* `export RAW_COLLECTION=LSSTComCam/raw/all`: The name of the raw data collection.
* `export CALIB_COLLECTIONS=LSSTComCam/calib/DM-46825`: Comma-separated list of curated or previously generated calibration collections to use as a starting point.
* `export TAG=newCalibs`: A human-readable tag to help indicate why a set of calibs were built (should also be findable from the ticket name).
* `export RERUN=20250122a`: The rerun name to indicate when the calibrations were generated.
* `export BOOTSTRAP_RUN_NUMBER=1`: The bootstrap run number ensures that the bootstrap run collections are unique in case of an initial mistake. (Because the run name for bootstraps are deterministic without a timestamp they cannot be rerun into the same collection). Note that all bootstrap calibs assume the same run number in the templates; if you modify just the dark (for example) you will need to edit the template.

* `export SELECTION_BIAS="instrument='LSSTComCam' and selection_string"`: The selection of raws to make the bootstrapBias and bias frames.
* `export SELECTION_DARK="instrument='LSSTComCam' and selection_string"`: The selection of raws to make the bootstrapDark and dark frames.
* `export SELECTION_FLAT_BOOTSTRAP="instrument='LSSTComCam' and selection_string"`: The selection of raws to make the bootstrapFlat.
* `export SELECTION_PTC="instrument='LSSTComCam' and selection_string"`: The selection of raws to generate the PTC.
* `export SELECTION_PTC_LINEARIZER=$SELECTION_PTC`: The selection of raws to generate the linearizer; usually will be the same as the PTC selection.
* `export SELECTION_PTC_BFK=$SELECTION_PTC`: The selection of raws to generate the brighter-fatter kernel; usually will be the same as the PTC selection.
* `export SELECTION_PTC_CTI=$SELECTION_PTC`: The selection of raws to generate the charge-transfer-inefficiency dataset; usually will be the same as the PTC selection.
* `export SELECTION_FLAT_g="instrument='LSSTComCam' and selection_string": The selection of raws to generate the g-band flat. See below for additional info.
* `export SELECTION_ILLUMINATION_CORRECTION="instrument='LSSTComCam' and selection_string": The selection of raws to generate illumination corrections. See below for additional info.
* `export SELECTION_GAIN_CORRECTION="instrument='LSSTCam' and selection_string": The selection of raw flat images to generate gain corrections.

Checking Environment Variables
------------------------------

Before submitting a bps file, it can be helpful to check that you have all the
correct environment variables set.  This can be mostly accomplished with the
following bash command (using PTC as an example):

```
cat $CP_PIPE_DIR/bps/templates/bps_ptc.yaml | envsubst
```

This will render all the env vars as bash would render them. Unfortunately, bps
uses python `os.path.expandvars()` which is not as sophisticated. The most
relevant limitation is that `envsubst` will replace an unset env var with a
empty string, while `expandvars()` will leave that raw
(e.g. `${USER_CALIB_PREFIX}` can be left verbatim). Nevertheless, this is a
useful sanity check before submission.


A Note on Flat Generation
-------------------------

Due to limitations in BPS environment variable expansion, we have one template
file for each band. There are template files for each of ugrizy. Note that it is
**strongly** recommended that the relevant physical filter be used in the
`selection_string` constraint on each flat selection.


A Note on Illumination Correction and Gain Correction Generation
--------------------------------------------

The illumination corrections and gain corrections depend on the full stack of previously generated calibrations.
The bps template here assumes that these are being run after a full calibration certification process.
Therefore, any calibrations used for this generation need to be explicitly put into the `CALIB_COLLECTIONS` environment variable.
