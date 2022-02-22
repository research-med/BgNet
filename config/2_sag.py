_base_ = ['./base/comm_config.py']

data = dict(
    train=dict(dataset=dict(dataset=dict(positions=['sagittal']))),
    val=dict(positions=['sagittal']),
    test=dict(positions=['sagittal']))