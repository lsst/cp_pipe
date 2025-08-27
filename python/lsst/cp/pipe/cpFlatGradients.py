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

import lsst.afw.cameraGeom
import lsst.pipe.base as pipeBase
from lsst.ip.isr import FlatGradient
import lsst.pex.config as pexConfig
from lsst.utils.plotting import make_figure

from .utils import FlatGradientFitter, bin_focal_plane

__all__ = [
    "CpFlatFitGradientsTask",
    "CpFlatFitGradientsConfig",
    "CpFlatApplyGradientsTask",
    "CpFlatApplyGradientsConfig",
]


class CpFlatFitGradientsConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "physical_filter"),
):
    camera = pipeBase.connectionTypes.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    input_flats = pipeBase.connectionTypes.Input(
        name="flat_uncorrected",
        doc="Input flats to fit gradients.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter"),
        multiple=True,
        deferLoad=True,
        isCalibration=True,
    )
    input_defects = pipeBase.connectionTypes.PrerequisiteInput(
        name="defects",
        doc="Input defect tables.",
        storageClass="Defects",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        multiple=True,
        deferLoad=True,
    )
    output_reference_gradient = pipeBase.connectionTypes.Output(
        name="flat_gradient_reference",
        doc="Reference flat gradient calibration.",
        storageClass="IsrCalib",
        dimensions=("instrument", "physical_filter"),
        isCalibration=True,
    )
    output_gradient = pipeBase.connectionTypes.Output(
        name="flat_gradient",
        doc="Flat gradient fit.",
        storageClass="IsrCalib",
        dimensions=("instrument", "physical_filter"),
    )
    model_residual_plot = pipeBase.connectionTypes.Output(
        name="gradient_model_residual_plot",
        doc="Residual plot from flat gradient model.",
        storageClass="Plot",
        dimensions=("instrument", "physical_filter"),
    )
    radial_model_plot = pipeBase.connectionTypes.Output(
        name="gradient_radial_model_plot",
        doc="Radial model plot for flat-field gradient.",
        storageClass="Plot",
        dimensions=("instrument", "physical_filter"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.do_reference_gradient:
            del self.output_gradient
        else:
            del self.output_reference_gradient


class CpFlatFitGradientsConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CpFlatFitGradientsConnections,
):
    do_reference_gradient = pexConfig.Field(
        dtype=bool,
        doc="Use task to produce a reference gradient (from sky)? This is used to "
            "control the dataset type of the output.",
        default=False,
    )
    bin_factor = pexConfig.Field(
        dtype=int,
        doc="Binning factor for flats going into the focal plane.",
        default=128,
    )
    do_constrain_zero = pexConfig.Field(
        dtype=bool,
        doc="Constrain the outermost radial spline value to zero?",
        default=True,
    )
    do_fit_centroid = pexConfig.Field(
        dtype=bool,
        doc="Fit a centroid offset from the focal plane centroid?",
        default=False,
    )
    do_fit_gradient = pexConfig.Field(
        dtype=bool,
        doc="Fit a linear gradient over the focal plane?",
        default=False,
    )
    do_fit_outer_gradient = pexConfig.Field(
        dtype=bool,
        doc="Fit a separate gradient to the outer region of the focal plane?",
        default=False,
    )
    do_normalize_center = pexConfig.Field(
        dtype=bool,
        doc="Normalize center of focal plane to 1.0?",
        default=True,
    )
    normalize_center_radius = pexConfig.Field(
        dtype=float,
        doc="Center normalization will be done using the average within this radius (mm).",
        default=25.0,
    )
    fp_centroid_x = pexConfig.Field(
        dtype=float,
        doc="Focal plane centroid x (mm).",
        default=0.0,
    )
    fp_centroid_y = pexConfig.Field(
        dtype=float,
        doc="Focal plane centroid y (mm).",
        default=0.0,
    )
    outer_gradient_radius = pexConfig.Field(
        dtype=float,
        doc="Minimum radius (mm) for the outer gradient fit.",
        default=325.0,
    )
    radial_spline_nodes = pexConfig.ListField(
        dtype=float,
        doc="Spline nodes to use for radial fit.",
        default=[0., 200., 250., 300., 310., 320., 330., 340., 350., 360., 368.],
    )
    min_flat_value = pexConfig.Field(
        dtype=float,
        doc="Minimum (relative) flat value to use in fit.",
        default=0.05,
    )
    max_flat_value = pexConfig.Field(
        dtype=float,
        doc="Maximum (relative) flat value to use in fit.",
        default=1.5,
    )
    detector_boundary = pexConfig.Field(
        dtype=int,
        doc="Do not use pixels within detector_boundary of the edge for fitting.",
        default=10,
    )
    fit_eps = pexConfig.Field(
        dtype=float,
        doc="Minimizer epsilon parameter.",
        default=1e-8,
    )
    fit_gtol = pexConfig.Field(
        dtype=float,
        doc="Minimizer gtol parameter.",
        default=1e-10,
    )


class CpFlatFitGradientsTask(pipeBase.PipelineTask):
    """Task to measure gradients on sky/dome flats.
    """

    ConfigClass = CpFlatFitGradientsConfig
    _DefaultName = "cpFlatFitGradients"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        input_flats = inputs["input_flats"]
        input_defects = inputs["input_defects"]

        input_flat_handle_dict = {
            handle.dataId["detector"]: handle for handle in input_flats
        }
        input_defect_handle_dict = {
            handle.dataId["detector"]: handle for handle in input_defects
        }

        struct = self.run(
            camera=inputs["camera"],
            input_flat_handle_dict=input_flat_handle_dict,
            input_defect_handle_dict=input_defect_handle_dict,
        )

        butlerQC.put(struct, outputRefs)

    def run(self, *, camera, input_flat_handle_dict, input_defect_handle_dict, binned_image=None):
        """Run the CpFlatFitGradientsTask.

        This task will fit full focal-plane gradients. See
        `lsst.cp.pipe.utils.FlatGradientFitter` for details.

        The return struct will contain ``output_gradient`` if
        do_reference_gradient is False, or ``output_reference_gradient``
        otherwise.

        Parameters
        ----------
        camera : `lsst.afw.cameraGeom.Camera`
            Camera object.
        input_flat_handle_dict : `dict` [`int`,
                                         `lsst.daf.butler.DeferredDatasetHandle`]
            Dictionary of input flat handles, keyed by detector.
        input_defect_handle_dict : `dict` [`int`,
                                           `lsst.daf.butler.DeferredDatasetHandle`]
            Dictionary of input defect handles, keyed by detector.
        binned_image : `np.ndarray`, optional
            Binned image from a previous run of the task. This will take
            precedence over the input handles, for easy re-use.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            Output structure with:
                ``output_gradient``: `lsst.ip.isr.FlatGradient` or
                ``output_reference_gradient``: `lsst.ip.isr.FlatGradient`
                ``model_residual_plot``: `matplotlib.Figure`
                ``radial_model_plot``: `matplotlib.Figure`
                ``binned_image``: `np.ndarray`
        """
        # Load in and bin the data.
        if binned_image is None:
            self.log.info("Loading and binning %d flats.", len(input_flat_handle_dict))
            binned = bin_focal_plane(
                input_flat_handle_dict,
                self.config.detector_boundary,
                self.config.bin_factor,
                defect_handle_dict=input_defect_handle_dict,
                include_itl_flag=True,
            )
        else:
            self.log.info("Using provided binned flat image.")
            binned = binned_image

        binned_unaltered = binned.copy()

        # Renormalize and filter out bad data.
        fp_radius = np.sqrt(
            (binned["xf"] - self.config.fp_centroid_x)**2.
            + (binned["yf"] - self.config.fp_centroid_y)**2.
        )
        use = (fp_radius < self.config.normalize_center_radius)
        central_value = np.median(binned["value"][use])

        if self.config.do_normalize_center:
            normalization = central_value
            binned["value"] /= normalization
            value_min = self.config.min_flat_value
            value_max = self.config.max_flat_value
        else:
            normalization = 1.0
            value_min = self.config.min_flat_value * central_value
            value_max = self.config.max_flat_value * central_value

        good = (
            np.isfinite(binned["value"])
            & (binned["value"] >= value_min)
            & (binned["value"] <= value_max)
            & (fp_radius <= self.config.radial_spline_nodes[-1])
        )
        binned = binned[good]

        # Do the fit.
        self.log.info("Fitting gradient to binned flat data.")
        nodes = self.config.radial_spline_nodes

        fitter = FlatGradientFitter(
            nodes,
            binned["xf"],
            binned["yf"],
            binned["value"],
            np.where(binned["itl"])[0],
            constrain_zero=self.config.do_constrain_zero,
            fit_centroid=self.config.do_fit_centroid,
            fit_gradient=self.config.do_fit_gradient,
            fit_outer_gradient=self.config.do_fit_outer_gradient,
            outer_gradient_radius=self.config.outer_gradient_radius,
            fp_centroid_x=self.config.fp_centroid_x,
            fp_centroid_y=self.config.fp_centroid_y,
        )
        p0 = fitter.compute_p0()
        pars = fitter.fit(p0, fit_eps=self.config.fit_eps, fit_gtol=self.config.fit_gtol)

        # Create the output FlatGradient calibration.
        gradient = FlatGradient()

        if "itl_ratio" in fitter.indices:
            itl_ratio = pars[fitter.indices["itl_ratio"]]
        else:
            itl_ratio = 1.0

        if self.config.do_fit_centroid:
            centroid_delta_x, centroid_delta_y = pars[fitter.indices["centroid_delta"]]
        else:
            centroid_delta_x, centroid_delta_y = 0.0, 0.0

        if self.config.do_fit_gradient:
            gradient_x, gradient_y = pars[fitter.indices["gradient"]]
        else:
            gradient_x, gradient_y = 0.0, 0.0

        if self.config.do_fit_outer_gradient:
            outer_gradient_x, outer_gradient_y = pars[fitter.indices["outer_gradient"]]
        else:
            outer_gradient_x, outer_gradient_y = 0.0, 0.0

        gradient.setParameters(
            radialSplineNodes=nodes,
            radialSplineValues=pars[fitter.indices["spline"]],
            itlRatio=itl_ratio,
            centroidX=self.config.fp_centroid_x,
            centroidY=self.config.fp_centroid_y,
            centroidDeltaX=centroid_delta_x,
            centroidDeltaY=centroid_delta_y,
            gradientX=gradient_x,
            gradientY=gradient_y,
            outerGradientX=outer_gradient_x,
            outerGradientY=outer_gradient_y,
            outerGradientRadius=self.config.outer_gradient_radius,
            normalizationFactor=normalization,
        )

        flat = input_flat_handle_dict[list(input_flat_handle_dict.keys())[0]].get()

        self.log.info("Making QA plots.")
        plot_dict = self._make_qa_plots(binned, gradient, flat.getFilter())

        # Set the calib metadata.
        filter_label = flat.getFilter()

        gradient.updateMetadata(camera=camera, filterName=filter_label.physicalLabel)
        gradient.updateMetadata(setDate=True, setCalibId=True)

        struct = pipeBase.Struct(
            model_residual_plot=plot_dict["model_residuals"],
            radial_model_plot=plot_dict["radial"],
            binned_image=binned_unaltered,
        )
        if self.config.do_reference_gradient:
            struct.output_reference_gradient = gradient
        else:
            struct.output_gradient = gradient

        return struct

    def _make_qa_plots(self, binned, gradient, filter_label):
        """Make QA plots for the binned data.

        Parameters
        ----------
        binned : `np.ndarray`
            Array with binned flat data.
        gradient : `lsst.ip.isr.FlatGradient`
            Flat gradient parameters.
        filter_label : `lsst.afw.image.FilterLabel`
            Filter label for labeling the plot.

        Returns
        -------
        plot_dict : `dict` [`str`, `matplotlib.Figure`]
            Dictionary of plot figures, keyed by name.
        """
        vmin, vmax = np.nanpercentile(binned["value"], [5, 95])
        model = gradient.computeFullModel(binned["xf"], binned["yf"], binned["itl"])

        # Fig1 is a 3 panel plot of data, model, and data/model.
        fig1 = make_figure(figsize=(16, 6))

        ax1 = fig1.add_subplot(131)

        im1 = ax1.hexbin(binned["xf"], binned["yf"], C=binned["value"], vmin=vmin, vmax=vmax)

        ax1.set_xlabel("Focal Plane x (mm)")
        ax1.set_ylabel("Focal Plane y (mm)")
        ax1.set_aspect("equal")
        ax1.set_title("Data")

        fig1.colorbar(im1, ax=ax1)

        ax2 = fig1.add_subplot(132)

        im2 = ax2.hexbin(binned["xf"], binned["yf"], C=model, vmin=vmin, vmax=vmax)

        ax2.set_xlabel("Focal Plane x (mm)")
        ax2.set_ylabel("Focal Plane y (mm)")
        ax2.set_aspect("equal")
        ax2.set_title("Model")

        fig1.colorbar(im2, ax=ax2)

        ax3 = fig1.add_subplot(133)

        ratio = binned["value"] / model
        vmin, vmax = np.nanpercentile(ratio, [1, 99])

        im3 = ax3.hexbin(binned["xf"], binned["yf"], C=ratio, vmin=vmin, vmax=vmax)

        ax3.set_xlabel("Focal Plane x (mm)")
        ax3.set_ylabel("Focal Plane y (mm)")
        ax3.set_aspect("equal")
        ax3.set_title("Data / Model")

        fig1.colorbar(im3, ax=ax3)

        fig1.suptitle(f"{filter_label.physicalLabel}: {self.config.connections.input_flats}")

        # Fig2 is a plot of the adjusted radial plot.
        fig2 = make_figure(figsize=(8, 6))
        ax = fig2.add_subplot(111)

        centroid_x = gradient.centroidX + gradient.centroidDeltaX
        centroid_y = gradient.centroidY + gradient.centroidDeltaY

        radius = np.sqrt((binned["xf"] - centroid_x)**2. + (binned["yf"] - centroid_y)**2.)
        value_adjusted = binned["value"].copy()

        value_adjusted *= gradient.computeGradientModel(binned["xf"], binned["yf"])
        value_adjusted[binned["itl"]] /= gradient.itlRatio

        ax.hexbin(radius, value_adjusted, bins="log")
        xvals = np.linspace(gradient.radialSplineNodes[0], gradient.radialSplineNodes[-1], 1000)
        yvals = gradient.computeRadialSplineModel(xvals)
        ax.plot(xvals, yvals, "r-")
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Value (adjusted)")
        ax.set_title(f"{filter_label.physicalLabel}: {self.config.connections.input_flats}")

        plot_dict = {
            "model_residuals": fig1,
            "radial": fig2,
        }

        return plot_dict


class CpFlatApplyGradientsConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "physical_filter", "detector"),
):
    camera = pipeBase.connectionTypes.PrerequisiteInput(
        name="camera",
        doc="Camera Geometry definition.",
        storageClass="Camera",
        dimensions=("instrument", ),
        isCalibration=True,
    )
    input_flat = pipeBase.connectionTypes.Input(
        name="flat_uncorrected",
        doc="Input flat to apply gradient correction.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )
    reference_gradient = pipeBase.connectionTypes.PrerequisiteInput(
        name="flat_gradient_reference",
        doc="Reference flat gradient.",
        storageClass="IsrCalib",
        dimensions=("instrument", "physical_filter"),
        isCalibration=True,
    )
    gradient = pipeBase.connectionTypes.Input(
        name="flat_gradient",
        doc="Flat gradient fit to full focal plane.",
        storageClass="IsrCalib",
        dimensions=("instrument", "physical_filter"),
    )
    output_flat = pipeBase.connectionTypes.Output(
        name="flat",
        doc="Output gradient-corrected flat.",
        storageClass="ExposureF",
        dimensions=("instrument", "detector", "physical_filter"),
        isCalibration=True,
    )


class CpFlatApplyGradientsConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=CpFlatApplyGradientsConnections,
):
    pass


class CpFlatApplyGradientsTask(pipeBase.PipelineTask):
    """Task to apply/remove gradients for dome flats.
    """

    ConfigClass = CpFlatApplyGradientsConfig
    _DefaultName = "cpFlatApplyGradients"

    def run(self, *, camera, input_flat, reference_gradient, gradient):
        """Run the CpFlatApplyGradientsTask.

        This will apply (remove) any gradients from a flat.

        Parameters
        ----------
        camera : `lsst.afw.cameraGeom.Camera`
            Camera object.
        input_flat : `lsst.afw.Exposure`
            Input flat to apply/remove gradients.
        reference_gradient : `lsst.ip.isr.FlatGradient`
            Reference gradient with target radial function.
        gradient : `lsst.ip.isr.FlatGradient`
            Gradient fit to the full focal plane.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            Output structure with:
                ``output_flat``: `lsst.afw.image.Exposure`
        """
        output_flat = input_flat.clone()

        detector = output_flat.getDetector()

        # Convert pixels to focal plane coordinates.
        xx = np.arange(output_flat.image.array.shape[1], dtype=np.int64)
        yy = np.arange(output_flat.image.array.shape[0], dtype=np.int64)
        x, y = np.meshgrid(xx, yy)
        x = x.ravel()
        y = y.ravel()

        transform = detector.getTransform(lsst.afw.cameraGeom.PIXELS, lsst.afw.cameraGeom.FOCAL_PLANE)
        xy = np.vstack((x, y))
        xf, yf = np.vsplit(transform.getMapping().applyForward(xy.astype(np.float64)), 2)
        xf = xf.ravel()
        yf = yf.ravel()

        # First we want to divide out the planar gradients.
        gradient_values = gradient.computeGradientModel(xf, yf)
        output_flat.image.array[y, x] *= gradient_values

        # Next we need the relative radial scaling.
        radial_values = (
            gradient.computeRadialSplineModelXY(xf, yf)
            / reference_gradient.computeRadialSplineModelXY(xf, yf)
        )
        output_flat.image.array[y, x] /= radial_values

        # And apply the normalization
        output_flat.image.array[:, :] /= gradient.normalizationFactor

        struct = pipeBase.Struct(output_flat=output_flat)

        return struct
