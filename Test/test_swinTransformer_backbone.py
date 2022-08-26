'''
This is a test to load a model with swin-transformer backbone!
by zhangbo 2022.4.1
the command for test is :
python ./Test/test_swinTransformer_backbone.py  configs/video/camvid/tmanet_r50-d8_640x640_80k_camvid_video.py
'''
import mmcv
import torch
from mmseg.models import build_segmentor
import argparse
from mmcv.utils import Config
def parse_args():
    parser = argparse.ArgumentParser(description='Test backbone')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args
args = parse_args()
cfg = Config.fromfile(args.config)
model = build_segmentor(cfg.model)
model=model.cuda(0)
'''
input:
    a tensor with the shape (1,3,2048,1024)
output:
    a list of tensor.(output from different level of the backbone)
    length is 4
    stage0 output: torch.Size([1, 128, 512, 256])
    stage1 output: torch.Size([1, 256, 256, 128])
    stage2 output: torch.Size([1, 512, 128, 64])
    stage3 output: torch.Size([1, 1024, 64, 32])
every stage? how to confusion the different level ouput of the backbone?
for confusing.
there are two solutions
one is resize contact and with the same encoding layer  advantage: without consider the shape of input 
the other is design different encoding layer            advantage: need to consider the shape of input!
'''
# img=torch.randn(1,3,2048,1024)
# print(len(model.extract_feat(img)))
# print(model.extract_feat(img)[0].shape)
# print(model.extract_feat(img)[1].shape)
# print(model.extract_feat(img)[2].shape)
sequence_imgs=torch.randn(2,2,3,512,256)
key_image=torch.randn(2,3,512,256)
key_image=key_image.cuda(0)
sequence_imgs=sequence_imgs.cuda(0)
sequence_imgs = [model.extract_feat(img) for img in sequence_imgs]
key_image=model.extract_feat(key_image)
output=model.decode_head(key_image,sequence_imgs)
print('output: ',output.shape)
