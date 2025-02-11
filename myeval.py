
from monai.metrics import DiceMetric, MeanIoU, SurfaceDiceMetric, SSIMMetric, GeneralizedDiceScore
import argparse
import os
from os.path import join

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from misc import get_idle_gpu, set_randomness


from cat_sam.datasets.whu import WHUDataset
from cat_sam.datasets.kvasir import KvasirDataset_test
from cat_sam.datasets.sbu import SBUDataset
from cat_sam.models.modeling import CATSAMT, CATSAMA
from cat_sam.utils.evaluators import SamHQIoU, StreamSegMetrics
from utils import get_dif
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
from utils import get_dif,test_save

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
                dice,gd,iou = test_save(test_dataloader,model,device,test_args.save_path)
            else:
                dice,gd,iou = test_save(test_dataloader,model,device)
            logging.info(f'Mean val dice: {dice}')
            logging.info(f'Mean val gd: {gd}')
            logging.info(f'Mean val iou: {iou}')
            
            print(f'Mean val dice: {dice}')
            print(f'Mean val gd: {gd}')
            print(f'Mean val iou: {iou}')

    else:
        test_dataloader = test_dl
        logging.info(f'Testing on {dataset} dataset...')
        print(f'Testing on {dataset} dataset...')
        if test_args.save_path is not None:
                dice,gd,iou = test_save(test_dataloader,model,device,test_args.save_path)
        else:
            dice,gd,iou = test_save(test_dataloader,model,device)
        logging.info(f'Mean val dice: {dice}')
        logging.info(f'Mean val gd: {gd}')
        logging.info(f'Mean val iou: {iou}')
            
        print(f'Mean val dice: {dice}')
        print(f'Mean val gd: {gd}')
        print(f'Mean val iou: {iou}')