'''
Create by zhangbo 2022.4.1
add decode_head 'tempo-spatial head' for new network!
'''
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from mmcv.cnn import ConvModule
from mmcv.cnn import ConvTranspose2d
from .decode_head import BaseDecodeHead
from ..builder import HEADS
from ..utils import SequenceConv
import torchsnooper
# def initialize_weights(*models):
#     """
#     Initialize Model Weights
#     """
#     for model in models:
#         for module in model.modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 nn.init.kaiming_normal_(module.weight)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#             elif isinstance(module, cfg.MODEL.BNFUNC):
#                 module.weight.data.fill_(1)
#                 module.bias.data.zero_()
class AttentionHead(nn.Module):
    def __init__(self,in_chs,out_chs):
        super(AttentionHead,self).__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        self.conv1=ConvModule(in_channels=in_chs,out_channels=out_chs,kernel_size=3
        ,stride=1,padding=1,norm_cfg=dict(type='BN', requires_grad=True),act_cfg=dict(type='ReLU6'))
        self.conv2=ConvModule(in_channels=out_chs,out_channels=out_chs,kernel_size=3
        ,stride=1,padding=1,norm_cfg=dict(type='BN', requires_grad=True),act_cfg=dict(type='ReLU6'))
        self.conv3=ConvModule(in_channels=out_chs,out_channels=out_chs,kernel_size=1
        ,stride=1,padding=0,norm_cfg=dict(type='BN', requires_grad=True),act_cfg=dict(type='ReLU6'))
        self.sig=nn.Sigmoid()
    def forward(self,x):
        x1=self.conv1(x)
        x1=self.conv2(x1)
        x1=self.conv3(x1)
        x1=self.sig(x1)
        return x1
class MemoryModule(nn.Module):
    """Memory read module.
    Args:

    """

    def __init__(self,
                 matmul_norm=False):
        super(MemoryModule, self).__init__()
        self.matmul_norm = matmul_norm

    def forward(self, memory_keys, memory_values, query_key, query_value):
        """
        Memory Module forward.
        Args:
            memory_keys (Tensor): memory keys tensor, shape: TxBxCxHxW
            memory_values (Tensor): memory values tensor, shape: TxBxCxHxW
            query_key (Tensor): query keys tensor, shape: BxCxHxW
            query_value (Tensor): query values tensor, shape: BxCxHxW

        Returns:
            Concat query and memory tensor.
        """
        sequence_num, batch_size, key_channels, height, width = memory_keys.shape
        _, _, value_channels, _, _ = memory_values.shape
        assert query_key.shape[1] == key_channels and query_value.shape[1] == value_channels
        memory_keys = memory_keys.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_keys = memory_keys.view(batch_size, key_channels, sequence_num * height * width)  # BxCxT*H*W

        query_key = query_key.view(batch_size, key_channels, height * width).permute(0, 2, 1).contiguous()  # BxH*WxCk
        key_attention = torch.bmm(query_key, memory_keys)  # BxH*WxT*H*W
        if self.matmul_norm:
            key_attention = (key_channels ** -.5) * key_attention
        key_attention = F.softmax(key_attention, dim=-1)  # BxH*WxT*H*W

        memory_values = memory_values.permute(1, 2, 0, 3, 4).contiguous()  # BxCxTxHxW
        memory_values = memory_values.view(batch_size, value_channels, sequence_num * height * width)
        memory_values = memory_values.permute(0, 2, 1).contiguous()  # BxT*H*WxC
        memory = torch.bmm(key_attention, memory_values)  # BxH*WxC
        memory = memory.permute(0, 2, 1).contiguous()  # BxCxH*W
        memory = memory.view(batch_size, value_channels, height, width)  # BxCxHxW

        query_memory = torch.cat([query_value, memory], dim=1)
        return query_memory
@HEADS.register_module()
class TSMA_Head(BaseDecodeHead):
    # init function:
    '''
    sequence_num: the T*
    key_channels: from satge0-stage3,excepted a list
    value_channels: same as above!
    '''
    def __init__(self, sequence_num, key_channels, value_channels, **kwargs):
        super(TSMA_Head, self).__init__(**kwargs)
        self.sequence_num = sequence_num
        assert isinstance(self.in_channels,list)
        assert isinstance(key_channels,int)
        #resize stage1 output a feature map
        #in following way, feature map of stage1 can be half shape. and the channel will be same as stage2
        self.stage1_resize=ConvModule(in_channels=self.in_channels[0],out_channels=self.in_channels[1],
                                      kernel_size=3,stride=2,padding=1,conv_cfg=self.conv_cfg,
                                      norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        #double stage3 size  and make the channels be same as stage2
        self.stage3_resize = ConvTranspose2d(in_channels=self.in_channels[2], out_channels=self.in_channels[1],
                                        kernel_size=3, stride=2, padding=1,output_padding=1)
        self.query_key_conv = nn.Sequential(
            ConvModule(
                self.in_channels[1],
                key_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                key_channels,
                key_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )

        self.query_value_conv = nn.Sequential(
            ConvModule(
                self.in_channels[1],
                value_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
            ConvModule(
                value_channels,
                value_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            ),
        )
        self.memory_key_conv=nn.Sequential(
            SequenceConv(self.in_channels[1], key_channels, 1, self.sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),  # 1*1 CONV
            SequenceConv(key_channels, key_channels, 3, self.sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)  # 3*3 CONV
        )
        self.memory_value_conv=nn.Sequential(
            SequenceConv(self.in_channels[1], value_channels, 1, self.sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg),  # 1*1 CONV
            SequenceConv(value_channels, value_channels, 3, self.sequence_num,
                         self.conv_cfg, self.norm_cfg, self.act_cfg)  # 3*3 CONV
        )
        self.memory_module = MemoryModule(matmul_norm=False)
        self.bottleneck = ConvModule(
            value_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.alpha1=AttentionHead(3*512,512) #输入通道3*512 输出通道
        self.alpha2=AttentionHead(3*512,512)
        self.alpha3=AttentionHead(3*512,512)

        self.upSample=nn.Sequential(
            ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=3, stride=2, padding=1,output_padding=1),
            ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=3, stride=2, padding=1,output_padding=1),
            ConvTranspose2d(in_channels=512, out_channels=512,
                                        kernel_size=3, stride=2, padding=1,output_padding=1),
            # ConvTranspose2d(in_channels=512, out_channels=512,
            #                             kernel_size=3, stride=2, padding=1,output_padding=1),
                                    
        )
    #sequence_imgs is a list of tensor  each tensor if the output of  different level.
    #@torchsnooper.snoop()
    def forward(self, inputs, sequence_imgs):
        #output of key frame backbone
        self.input_transform='multiple_select'
        x = self._transform_inputs(inputs) #filter out unused layer...  a list of tensor
        #output of different level of backbone of every past frame
        x[0]=self.stage1_resize(x[0])
        if x[1].shape[3]%2==0:
            self.stage3_resize.output_padding=1
        else:
            self.stage3_resize.output_padding=0
        x[2]=self.stage3_resize(x[2])
        x=[xx.unsqueeze(0) for xx in x]
        # x shape is 3 2 512 49 49
        #x shape is 2 3*512 49 49
        #深度拷贝x 以防后面出现问题
        y=[xx.contiguous() for xx in x]
        y=torch.cat(y,dim=0) 
        y=y.permute(1,0,2,3,4)
        y=y.reshape(y.shape[0],y.shape[1]*y.shape[2],y.shape[3],y.shape[4])
        weights=[]
        weights.append(self.alpha1(y))
        weights.append(self.alpha2(y))
        weights.append(self.alpha3(y))
        # for p in self.alpha1.parameters():
        #     print(p.grad)
        x=x[0]*weights[0]+x[1]*weights[1]+x[2]*weights[2]
        x=x.squeeze(0)
        query_key=self.query_key_conv(x)
        query_value=self.query_value_conv(x)
        #sequence_imgs is 2 dimension list
        sequence_frames=[]
        for frame in sequence_imgs:
            frame=self._transform_inputs(frame) #3, B C H W
            frame[0]=self.stage1_resize(frame[0])
            frame[2]=self.stage3_resize(frame[2])
            frame=[xx.unsqueeze(0) for xx in frame]
            y=[xx.contiguous() for xx in frame]
            y=torch.cat(y,dim=0) 
            y=y.permute(1,0,2,3,4)
            y=y.reshape(y.shape[0],y.shape[1]*y.shape[2],y.shape[3],y.shape[4])
            weights=[]
            weights.append(self.alpha1(y))
            weights.append(self.alpha2(y))
            weights.append(self.alpha3(y))
            frame=frame[0]*weights[0]+frame[1]*weights[1]+frame[2]*weights[2]
            sequence_frames.append(frame)  #T,B C H W
        sequence_frames=torch.cat(sequence_frames,dim=0) #T B C H W
        memory_keys=self.memory_key_conv(sequence_frames)
        memory_values=self.memory_value_conv(sequence_frames)
        # memory read
        output = self.memory_module(memory_keys, memory_values, query_key, query_value)
        output = self.bottleneck(output)
        output=self.upSample(output)
        # print('output: ', output.shape)
        output = self.cls_seg(output)
        return output