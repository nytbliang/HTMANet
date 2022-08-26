import argparse
import os
import os.path as osp
import sys
import time
import torch
import mmcv
import timeit
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import build_dataloader, build_dataset
import operator
from mmseg.models import build_segmentor
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel
import time
def single_gpu_test(model, data_loader, show=False, out_dir=None, test_fps=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped
        into the directory to save output results.
        test_fps (str, optional): If True, to test model inference fps.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    total_time = 0.0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            start_time = timeit.default_timer()
            start=time.time()
            result = model(return_loss=False, **data)
            end=time.time()
            print("running time is ",end-start)

        if isinstance(result, list):
            results.extend(result)
        else:
            results.append(result)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results
def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    # parser.add_argument('--work-dir', help='the dir to save logs and models')
    # parser.add_argument(
    #     '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    # parser.add_argument('--out', help='output result file in pickle format')
    # parser.add_argument('--seed', type=int, default=None, help='random seed')
    # parser.add_argument(
    #     '--deterministic',
    #     action='store_true',
    #     help='whether to set deterministic options for CUDNN backend.')
    # parser.add_argument(
    #     '--format-only',
    #     action='store_true',
    #     help='Format the output results without perform evaluation. It is'
    #          'useful when you want to format the result to a specific format and '
    #          'submit it to the test server')
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
    #          ' for generic datasets, and "cityscapes" for Cityscapes')
    # parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument(
    #     '--show-dir', help='directory where painted images will be saved')
    # parser.add_argument(
    #     '--gpu-collect',
    #     action='store_true',
    #     help='whether to use gpu to collect results.')
    # parser.add_argument(
    #     '--tmpdir',
    #     help='tmp directory used for collecting results from multiple '
    #          'workers, available when gpu_collect is not specified')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
args = parse_args()
#载入配置文件
cfg = mmcv.Config.fromfile(args.config)
#载入数据集
dataset = build_dataset(cfg.data.val, dict(test_mode=True))
dataset.img_infos=sorted(dataset.img_infos, key=operator.itemgetter('filename'))
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
cfg.model.pretrained = None
cfg.data.test.test_mode = True
model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
model.CLASSES = checkpoint['meta']['CLASSES']
model.PALETTE = checkpoint['meta']['PALETTE']
model = MMDataParallel(model, device_ids=[0])
outputs = single_gpu_test(model, data_loader, False)
