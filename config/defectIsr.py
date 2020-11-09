# Set ISR processing to run up until we would be applying the defect
# correction.
config.isr.doWrite = True
config.isr.doOverscan = True
config.isr.doAssembleCcd = True
config.isr.doBias = True
config.isr.doVariance = False
config.isr.doLinearize = False
config.isr.doCrosstalk = False
config.isr.doBrighterFatter = False
config.isr.doDark = False
config.isr.doStrayLight = False
config.isr.doFlat = False
config.isr.doFringe = False
config.isr.doApplyGains = False
config.isr.doDefect = False
config.isr.doSaturationInterpolation = False
config.isr.growSaturationFootprintSize = 0
