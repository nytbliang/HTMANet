import os
import pickle
from unicodedata import name
import numpy as np
import cv2
import torch
naic_class = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],[190, 153, 153], [153, 153, 153],
           [250, 170, 30], [220, 220, 0],[107, 142, 35], [152, 251, 152], [70, 130, 180],
           [220, 20, 60],[255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
# naic_class = ('void', 'sky', 'building', 'column_pole', 'road',
#            'sidewalk', 'tree', 'sing_symbol', 'fence', 'car',
#            'pedestrian', 'bicyclist')

# palette = [[0, 0, 0], [128, 128, 128], [128, 0, 0], [192, 192, 128],
#            [128, 64, 128], [0, 0, 192], [128, 128, 0], [192, 128, 128],
#            [64, 64, 128], [64, 0, 128], [64, 64, 0], [0, 128, 192]]
preds=torch.load("./result.pth")
predNames=dict()
for pred in preds:
    videoID=pred['resultName'].split('_')[1]
    if videoID not in predNames:
        predNames[videoID]=[]
    else:
        predNames[videoID].append(pred['resultName'])
    color_seg = np.zeros((pred['pred'][0].shape[0], pred['pred'][0].shape[1]), dtype=np.uint8)
    label = pred['pred'][0].astype(np.uint8)
    # for label_id, color in enumerate(palette):
    #     color_seg[label == label_id, :] = color
    #     color_seg = color_seg[..., ::-1]
    cv2.imwrite('./result/pred_result/'+pred['resultName'],pred['pred'][0].astype(np.uint8))
    print('has written ./result/'+pred['resultName'])
    #cv2.imshow('img',color_seg)
    #cv2.waitKey(0)
    #break
val_sam_dir="./result/val_sam"
txt_names=[]
for vid,names in predNames.items():
    txt_name=os.path.join(val_sam_dir,names[len(names)-1].replace('png','txt'))
    txt_names.append(names[len(names)-1].replace('png','txt'))
    with open(txt_name,'w') as f:
        for i in range(len(names)-1):
            f.write('leftImg8bit/val/lindau/'+names[i]+'\n')
with open('./result/val_video_list_sam.lst','w') as f:
    for temp in txt_names:
        f.write(temp+'\n')
