# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    #--
    # pretrained='./init_models/resnet50_v1c-2cccc1ad.pth',
    # backbone=dict(
    #     type='ResNetV1c',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     dilations=(1, 1, 2, 4),
    #     strides=(1, 2, 1, 1),
    #     norm_cfg=norm_cfg,
    #     norm_eval=False,
    #     style='pytorch',
    #     contract_dilation=True),
    #++
    pretrained='init_models/swin_base_patch4_window12_384_22k.pth',#'./init_models/swin_base_patch4_window12_384.pth',
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_rate=0.3,
        attn_drop_rate=0.3,
        #pretrained='init_models/swin_base_patch4_window12_384_22k.pth'
    ),
    # decode_head=dict(
    #     type='TMAHead',
    #     in_channels=1024,#2048,
    #     in_index=3,
    #     channels=512,
    #     sequence_num=2,
    #     key_channels=256,
    #     value_channels=1024,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    decode_head=dict(
        type='TSMA_Head',
        in_channels=[256,512,1024],
        in_index=[1,2,3],
        channels=512,
        sequence_num=2,
        key_channels=256,
        value_channels=1024,
        dropout_ratio=0.1,
        num_classes=19,
        input_transform='multiple_select',
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),

    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,#1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)))
# model training and testing settings
# 表示动态 scale
fp16 = dict(loss_scale=512.)
find_unused_parameters=True
#fp16 = dict(loss_scale=512.)#'dynamic')
train_cfg = dict()
test_cfg = dict(mode='whole')
