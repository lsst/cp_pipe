description: cp_pipe task to plot Photon Transfer Curve dataset
tasks:
    plotPtcTask:
        class: lsst.cp.pipe.ptc.PlotPhotonTransferCurveTask
        config:
            connections: inputPtcDataset
