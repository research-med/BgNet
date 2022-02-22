_base_ = ['./base/comm_config.py']

data = dict(
    train=dict(dataset=dict(dataset=dict(positions=['axial', 'sagittal']))),
    val=dict(positions=['axial', 'sagittal']),
    test=dict(positions=['axial', 'sagittal']))