# the config of feature layer fusion
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetST', arch='S', img_size=224, in_channels=3, drop_rate=0.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='BgNetLinearClsHead',
        num_classes=1,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        use_sigmod=True,
        loss=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0,
            use_sigmoid=True,
            class_weight=[0.2, 0.8]),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ])

# the config of input layer
train_pipeline = [
    dict(
        type='InputLyaerFusion',
        to_float32=True,
        target_size=(224, 224),
    ),
    dict(type='CustomNormalize'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label', 'patient_id'])
]

test_pipeline = [
    dict(
        type='InputLyaerFusion',
        test_mode=True,
        to_float32=True,
        target_size=(224, 224)),
    dict(type='CustomNormalize'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'patient_id'])
]

# =================== dataset ======================

train_json = '/path/to/your/train.json'
test_json = '/path/to/your/test.json'

# 数据集设置
data_root = '/path/to/your/data'
dataset_type = 'SpineDataset'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type=dataset_type,
            data_prefix=data_root,
            pipeline=train_pipeline,
            json_file=train_json,
            classes=['Benign', 'Malignant'],
            age_on_decision=True)),
    val=dict(
        type=dataset_type,
        data_prefix=data_root,
        json_file=test_json,
        classes=['Benign', 'Malignant'],
        pipeline=test_pipeline,
        test_mode=True,
        flag='valid',
        age_on_decision=True),
    test=dict(
        type=dataset_type,
        data_prefix=data_root,
        json_file=test_json,
        classes=['Benign', 'Malignant'],
        pipeline=test_pipeline,
        test_mode=True,
        flag='test',
        age_on_decision=True))

# log
log_level = 'INFO'
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# other config
dist_params = dict(backend='nccl')
load_from = None
resume_from = None
workflow = [('train', 1)]

optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# lr
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-3,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=2 * 78,
    warmup_by_epoch=False)

# epoch
runner = dict(type='EpochBasedRunner', max_epochs=20)

# evaluation frequency
evaluation = dict(interval=20)

# checkpoint frequency
checkpoint_config = dict(interval=20)

# prediction is made at the patient level
eval_by_patient = True

find_unused_parameters = True