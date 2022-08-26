_base_ = [
    '../../_base_/models/tmanet_r50-d8.py', '../../_base_/datasets/camvid_video.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# model = dict(
#     decode_head=dict(type='TMAHead', num_classes=12, sequence_num=4, key_channels=64, value_channels=256),
#     auxiliary_head=dict(num_classes=12)
# )
model = dict(
    decode_head=dict(type='TSMA_Head', num_classes=12,align_corners=True, sequence_num=2, key_channels=64, value_channels=256),
    auxiliary_head=dict(align_corners=True,num_classes=12))
data = dict(train=dict(sequence_range=2,
                       sequence_num=2),
            val=dict(sequence_range=2,
                     sequence_num=2),
            test=dict(sequence_range=2,
                      sequence_num=2))
test_cfg = dict(mode='slide', crop_size=(640, 640), stride=(512, 512))
