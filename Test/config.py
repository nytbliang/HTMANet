_base_ = []
backbone = dict(
    pretrained='./init_models/swin_base_patch4_window12_384.pth',
    type='SwinTransformer',
    pretrain_img_size=384,
    embed_dims=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=12)

