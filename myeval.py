
from monai.metrics import DiceMetric, MeanIoU, SurfaceDiceMetric, SSIMMetric, GeneralizedDiceScore
import argparse
import os
from os.path import join

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from misc import get_idle_gpu, set_randomness
from train import batch_to_cuda

from cat_sam.datasets.whu import WHUDataset
from cat_sam.datasets.kvasir import KvasirDataset_test
from cat_sam.datasets.sbu import SBUDataset
from cat_sam.models.modeling import CATSAMT, CATSAMA
from cat_sam.utils.evaluators import SamHQIoU, StreamSegMetrics
from cat_sam.utils import get_dif
import numpy as np
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', required=True, type=str, help="Path to the dataset config file")
    parser.add_argument('--model_config', required=True, type=str, help="Path to the model config file")
    parser.add_argument(
        '--log_dir', required=True, type=str, default=None,
        help=""
    )
    parser.add_argument(
        '--save_path', required=False, type=str, default=None,
        help=""
    )
    return parser.parse_args()

import logging


import torch.nn.functional as F
from cat_sam.build_model import build_model,build_dataloader_eval
import cv2
from PIL import Image as Image
from cat_sam.utils import get_diff

    
def test_save(test_dataloader,model,save_path=None):
    if save_path is not None and not os.path.exists(save_path):
        os.mkdir(save_path)
    
    with torch.no_grad():
        batch_dice = []
        batch_gd = []
        batch_iou = []
        name = 0
        for test_step, batch in enumerate(tqdm(test_dataloader)):
            batch = batch_to_cuda(batch, device)
            model.set_infer_img(img=batch['images'])

            masks_pred = model.infer(box_coords=batch['box_coords'])
            masks_gt = batch['gt_masks']
            
            for mask_pred, mask_gt in zip(masks_pred, masks_gt):
                mask_pred = mask_pred.cpu().numpy()
                mask_gt = mask_gt.cpu().numpy()
                gt = mask_gt
                
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                
                dice = DiceMetric()
                gd =  GeneralizedDiceScore()
                iou = MeanIoU()
                
                res = torch.tensor(mask_pred)
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = res > 0.5
                res = torch.tensor(res).reshape(1,1,res.shape[0],res.shape[1])
                gt = torch.tensor(gt).reshape(1,1,gt.shape[0],gt.shape[1])
                dice(res, gt)
                gd(res, gt)
                iou(res, gt)
                final_dice = dice.aggregate().numpy()[0]
                final_gd = gd.aggregate().numpy()[0]
                final_iou = iou.aggregate().numpy()[0]
                batch_dice.append(final_dice)
                batch_gd.append(final_gd)
                batch_iou.append(final_iou)
                res = res.squeeze().cpu().numpy()
                res = np.round(res * 255).astype(np.uint8)
                gt = gt.squeeze().cpu().numpy()
                gt = np.round(gt * 255).astype(np.uint8)
                if save_path is not None:
                    diff = get_dif(gt,res)
                    
                    diff.save(os.path.join(save_path, str(name)+".png"))
                    name += 1
        logging.info(f'Mean val dice: {sum(batch_dice) / len(batch_dice)}')
        logging.info(f'Mean val gd: {sum(batch_gd) / len(batch_gd)}')
        logging.info(f'Mean val iou: {sum(batch_iou) / len(batch_iou)}')
        
        print(f'Mean val dice: {sum(batch_dice) / len(batch_dice)}')
        print(f'Mean val gd: {sum(batch_gd) / len(batch_gd)}')
        print(f'Mean val iou: {sum(batch_iou) / len(batch_iou)}')
    print('Test Done!')
test_args = parse()
used_gpu = get_idle_gpu(gpu_num=1)
if test_args.log_dir is not None and not os.path.exists(test_args.log_dir):
    os.mkdir(test_args.log_dir)
if test_args.save_path is not None and not os.path.exists(test_args.save_path):
    os.mkdir(test_args.save_path)
logging.basicConfig(filename=os.path.join(test_args.log_dir,'eval.log'), format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu[0])
from cat_sam.config import load_config
if __name__ == '__main__':
    set_randomness()
    logging.info(f'Now using GPU: {used_gpu[0]}')
    logging.info(f'Arguments: {test_args}')
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_config = load_config(test_args.model_config)
    dataset_config = load_config(test_args.dataset_config)
    dataset = dataset_config['dataset']
    model = build_model(model_config,device)
    test_dl = build_dataloader_eval(dataset_config)

    model = build_model(test_args)
    if dataset == 'divide':
        test_datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
        
        for i,dataset in enumerate(test_datasets):
            test_dataloader = test_dl[i]
            logging.info(f'Testing on {dataset} dataset...')
            print(f'Testing on {dataset} dataset...')
            if test_args.save_path is not None:
                test_save(test_dataloader,model,test_args.save_path)
            else:
                test_save(test_dataloader,model)
    else:
        test_dataloader = test_dl
        logging.info(f'Testing on {dataset} dataset...')
        print(f'Testing on {dataset} dataset...')
        if test_args.save_path is not None:
            test_save(test_dataloader,model,test_args.save_path)
        else:
            test_save(test_dataloader,model)