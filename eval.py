
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

import numpy as np
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', default='./data', type=str,
        help="The directory that the datasets are placed. Default to be ./data"
    )
    parser.add_argument(
        '--num_workers', default=4, type=int,
        help="The num_workers argument used for the testing dataloaders. Default to be 4."
    )
    parser.add_argument(
        '--batch_size', default=2, type=int,
        help="The batch size for the testing dataloader. Default to be 2."
    )
    parser.add_argument(
        '--dataset', required=True, type=str, choices=['whu', 'sbu', 'kvasir','divide'],
        help="Your target dataset. This argument is required."
    )
    parser.add_argument(
        '--ckpt_path', required=True, type=str,
        help="The absolute path to your target checkpoint file. This argument is required."
    )
    parser.add_argument(
        '--sam_type', default='vit_l', type=str, choices=['vit_b', 'vit_l', 'vit_h','rein_vit_h','rein_vit_l','rein_vit_b'],
        help='The type of the backbone SAM model. Default to be vit_l.'
    )
    parser.add_argument(
        '--cat_type', required=True, type=str, choices=['cat-a', 'cat-t'],
        help='The type of the CAT-SAM model. This argument is required.'
    )
    parser.add_argument(
        '--rein_type', required=True, type=str, default=None,
        help="the type of rein"
    )
    return parser.parse_args()

import logging
logging.basicConfig(filename='./eval_log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

import torch.nn.functional as F
def build_model(worker_args):
    model = None
    reins_config = None
    if worker_args.sam_type in ['rein_vit_l','rein__vit_h']:
        reins_config=dict(
            token_length=100,
            link_token_to_query=True,
            lora_dim=16,
            zero_mlp_delta_f=False,  # v2
        )
        from cat_sam.models.segment_anything_ext import change_rein_cfg
        reins_config = change_rein_cfg(model_type=worker_args.sam_type,rein_cfg=reins_config)

    assert worker_args.cat_type in ['cat-a', 'cat-t']
    assert worker_args.ckpt_path is not None
    if worker_args.cat_type == 'cat-t':
        model_class = CATSAMT
    elif worker_args.cat_type == 'cat-a':
        model_class = CATSAMA
    else:
        raise ValueError(f'invalid cat_type: {worker_args.cat_type}!')
    model = model_class(model_type=worker_args.sam_type,rein_cfg=reins_config).to(device=device)
    model_state_dict = torch.load(worker_args.ckpt_path, map_location=device)
    if 'model' in model_state_dict.keys():
        model_state_dict = model_state_dict['model']
    model.load_state_dict(model_state_dict)
    return model
    
def test(test_dataloader,model):
    
    with torch.no_grad():
        batch_dice = []
        batch_gd = []
        batch_iou = []
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


                # print(res.shape,gt.shape)
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = res > 0.5
                # print(res.shape,gt.shape)
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
                # print('save img to: ', save_path + '/'+ name)
                res = res.squeeze().cpu().numpy()
                res = np.round(res * 255).astype(np.uint8)
                # cv2.imwrite(os.path.join(save_path, name), res)
        logging.info(f'Mean val dice: {sum(batch_dice) / len(batch_dice)}')
        logging.info(f'Mean val gd: {sum(batch_gd) / len(batch_gd)}')
        logging.info(f'Mean val iou: {sum(batch_iou) / len(batch_iou)}')
        
        print(f'Mean val dice: {sum(batch_dice) / len(batch_dice)}')
        print(f'Mean val gd: {sum(batch_gd) / len(batch_gd)}')
        print(f'Mean val iou: {sum(batch_iou) / len(batch_iou)}')
    print('Test Done!')



test_args = parse()
used_gpu = get_idle_gpu(gpu_num=1)
os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu[0])
test_args.used_gpu, test_args.gpu_num = used_gpu, 1

if __name__ == '__main__':
    set_randomness()
    logging.info(f'Now using GPU: {used_gpu[0]}')
    logging.info(f'Arguments: {test_args}')
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dataset_class = KvasirDataset_test

    model = build_model(test_args)
    if test_args.dataset == 'divide':
        test_datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
        # test_datasets = ['Kvasir']
        for dataset in test_datasets:
            test_dataset = dataset_class(
                data_dir=join(test_args.data_dir, dataset)
            )
            test_dataloader = DataLoader(
                dataset=test_dataset, shuffle=False, drop_last=False,
                batch_size=test_args.batch_size, num_workers=test_args.num_workers,
                collate_fn=test_dataset.collate_fn
            )
            logging.info(f'Testing on {dataset} dataset...')
            print(f'Testing on {dataset} dataset...')

            test(test_dataloader,model)
    