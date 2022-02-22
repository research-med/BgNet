_base_ = ['./base/comm_config.py']

data = dict(
    train=dict(dataset=dict(dataset=dict(positions=['axial']))),
    val=dict(positions=['axial']),
    test=dict(positions=['axial']))