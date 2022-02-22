_base_ = ['./base/comm_config.py']

data = dict(
    train=dict(
        dataset=dict(
            dataset=dict(bgnet=True, positions=['axial', 'sagittal']))),
    val=dict(bgnet=True, positions=['axial', 'sagittal']),
    test=dict(bgnet=True, positions=['axial', 'sagittal']))
