# Set ISR processing to run up until we would be applying the CT
# correction.  Applying subsequent stages may corrupt the signal.
config.isr.doWrite = True
config.isr.doOverscan = True
config.isr.doAssembleCcd = True
config.isr.doBias = True
config.isr.doVariance = False  # This isn't used in the calculation below.
config.isr.doLinearize = True  # This is the last ISR step we need.
config.isr.doCrosstalk = False
config.isr.doBrighterFatter = False
config.isr.doDark = False
config.isr.doStrayLight = False
config.isr.doFlat = False
config.isr.doFringe = False
config.isr.doApplyGains = False
config.isr.doDefect = True  # Masking helps remove spurious pixels.
config.isr.doSaturationInterpolation = False
config.isr.growSaturationFootprintSize = 0  # We want the saturation spillover: it's good signal.
