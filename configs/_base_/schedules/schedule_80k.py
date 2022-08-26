# optimizer
optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=1.00, min_lr=1e-5, by_epoch=False)
# lr_config = dict(
#     policy='CosineAnnealingWarmRestarts',
#     warmup='linear',
#     warmup_iters=2000,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5)
lr_config = dict(
    policy='CosineRestart',
    restart_weights=[1,1,1,1,1],
    periods=[3750,7500,15000,30000,60000],
    min_lr=1e-6)
# lr_config = dict(
#     policy='CosineRestart',
#     restart_weights=[1,1,1,1,1],
#     periods=[3000,6000,12000,24000,48000],
#     min_lr=1e-6)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=120000)
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=20)
evaluation = dict(interval=2000, metric='mIoU')
