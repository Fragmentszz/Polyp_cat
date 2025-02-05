import argparse
from genericpath import exists
from json import load
from math import log
import os
import random
from contextlib import nullcontext
from functools import partial
from os.path import join

import torch.nn.functional as F
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from tqdm import tqdm
from misc import get_idle_gpu, get_idle_port, set_randomness

from cat_sam.datasets.whu import WHUDataset
from cat_sam.datasets.kvasir import KvasirDataset
from cat_sam.datasets.sbu import SBUDataset
from cat_sam.datasets.transforms import HorizontalFlip, VerticalFlip, RandomCrop
from cat_sam.models.modeling import CATSAMT, CATSAMA,Reins
from cat_sam.utils.evaluators import SamHQIoU, StreamSegMetrics
import logging
# from cat_sam.models.segment_anything_ext
# /applications/graduate_design/cat-sam/cat_sam/models/segment_anything_ext/build_sam.py
def calculate_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    assert inputs.size(0) == targets.size(0)
    inputs = inputs.sigmoid()
    inputs, targets = inputs.flatten(1), targets.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def worker_init_fn(worker_id: int, base_seed: int, same_worker_seed: bool = True):
    """
    Set random seed for each worker in DataLoader to ensure the reproducibility.

    """
    seed = base_seed if same_worker_seed else base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
from datetime import datetime



def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_dir', default='./exp', type=str,
        help="The directory to save the best checkpoint file. Default to be ./exp"
    )
    parser.add_argument(
        '--model_config',default=None, type=str, required=True, 
        help='the config file of model'
    )
    parser.add_argument(
        '--dataset_config',default=None, type=str, required=True, 
        help='the config file of train&eval dataset'
    )
    return parser.parse_args()


from utils import batch_to_cuda
from cat_sam.build_model import build_model, build_dataloader_train
from cat_sam.config import load_config



def main_worker(worker_id, worker_args):
    set_randomness()
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    
    
    model_config = load_config(worker_args.model_config)
    dataset_config = load_config(worker_args.dataset_config)
    gpu_num = len(worker_args.used_gpu)
    world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ.keys() else gpu_num
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = base_rank * gpu_num + worker_id
    if gpu_num > 1:
        dist.init_process_group(backend='nccl', init_method=worker_args.dist_url,
                                world_size=world_size, rank=local_rank)

    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)


    model = build_model(model_config,device,local_rank)
    
    
    model.train()



    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-2
    )
    max_epoch_num, valid_per_epochs = 20, 2

    train_dataloader,val_dataloader = build_dataloader_train(dataset_config,max_epoch_num,local_rank)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=max_epoch_num, eta_min=1e-5
    )

    best_miou = 0
    class_names = ['Background', 'Foreground']
    iou_eval = StreamSegMetrics(class_names=class_names)

    if not os.path.exists(worker_args.exp_dir):
        os.mkdir(worker_args.exp_dir)
    reins_config = model_config['reins_config']
    exp_path = join(
        worker_args.exp_dir,
        f'{reins_config['rein_type']}{'_evp_feature' if reins_config['if_evp_feature'] else ''}_{'4+1_layer' if reins_config['local_block'] else '4_layer'}{'_connect_hq_token' if reins_config['connect_hq_token'] else ''}'
    )
    os.makedirs(exp_path, exist_ok=True)
    
    for epoch in range(1, max_epoch_num + 1):
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        train_pbar = None
        if local_rank == 0:
            train_pbar = tqdm(total=len(train_dataloader), desc='train', leave=False)
        total_step = len(train_dataloader)
        model.train()
        for train_step, batch in enumerate(train_dataloader):
            batch = batch_to_cuda(batch, device)
            masks_pred = model(
                imgs=batch['images'], point_coords=batch['point_coords'], point_labels=batch['point_labels'],
                box_coords=batch['box_coords'], noisy_masks=batch['noisy_object_masks']
            )
            masks_gt = batch['object_masks']
            for masks in [masks_pred, masks_gt]:
                for i in range(len(masks)):
                    if len(masks[i].shape) == 2:
                        masks[i] = masks[i][None, None, :]
                    if len(masks[i].shape) == 3:
                        masks[i] = masks[i][:, None, :]
                    if len(masks[i].shape) != 4:
                        raise RuntimeError

            bce_loss_list, dice_loss_list, focal_loss_list = [], [], []
            for i in range(len(masks_pred)):
                pred, label = masks_pred[i], masks_gt[i]
                label = torch.where(torch.gt(label, 0.), 1., 0.)
                b_loss = F.binary_cross_entropy_with_logits(pred, label.float())
                d_loss = calculate_dice_loss(pred, label)

                bce_loss_list.append(b_loss)
                dice_loss_list.append(d_loss)

            bce_loss = sum(bce_loss_list) / len(bce_loss_list)
            dice_loss = sum(dice_loss_list) / len(dice_loss_list)
            total_loss = bce_loss + dice_loss
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                bce_loss=bce_loss.clone().detach(),
                dice_loss=dice_loss.clone().detach()
            )

            backward_context = nullcontext
            if torch.distributed.is_initialized():
                backward_context = model.no_sync
            with backward_context():
                total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if torch.distributed.is_initialized():
                for key in loss_dict.keys():
                    if hasattr(loss_dict[key], 'detach'):
                        loss_dict[key] = loss_dict[key].detach()
                    torch.distributed.reduce(loss_dict[key], dst=0, op=torch.distributed.ReduceOp.SUM)
                    loss_dict[key] /= torch.distributed.get_world_size()

            if train_pbar:
                train_pbar.update(1)
                str_step_info = "Epoch: {epoch}/{epochs:4}. " \
                                "Loss: {total_loss:.4f}(total), {bce_loss:.4f}(bce), {dice_loss:.4f}(dice)".format(
                    epoch=epoch, epochs=max_epoch_num,
                    total_loss=loss_dict['total_loss'], bce_loss=loss_dict['bce_loss'], dice_loss=loss_dict['dice_loss']
                )
                train_pbar.set_postfix_str(str_step_info)
            if train_step % 100 == 0 or train_step == 1 or train_step == total_step:
                logging.info(
                    '#TRAIN#:{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], total_loss: {:.4f}, '
                    'bce_loss: {:.4f},  dice_loss: {:.4f}'.
                    format(datetime.now(), epoch, max_epoch_num, train_step, total_step, 
                            loss_dict['total_loss'], loss_dict['bce_loss'], loss_dict['dice_loss']))
        scheduler.step()
        if train_pbar:
            train_pbar.clear()
        
        if local_rank == 0 and epoch % valid_per_epochs == 0:
            model.eval()
            valid_pbar = tqdm(total=len(val_dataloader), desc='valid', leave=False)
            for val_step, batch in enumerate(val_dataloader):
                batch = batch_to_cuda(batch, device)
                val_model = model
                if hasattr(model, 'module'):
                    val_model = model.module

                with torch.no_grad():
                    val_model.set_infer_img(img=batch['images'])
                    if worker_args.dataset == 'm_roads':
                        masks_pred = val_model.infer(point_coords=batch['point_coords'])
                    else:
                        masks_pred = val_model.infer(box_coords=batch['box_coords'])

                masks_gt = batch['gt_masks']
                for masks in [masks_pred, masks_gt]:
                    for i in range(len(masks)):
                        if len(masks[i].shape) == 2:
                            masks[i] = masks[i][None, None, :]
                        if len(masks[i].shape) == 3:
                            masks[i] = masks[i][None, :]
                        if len(masks[i].shape) != 4:
                            raise RuntimeError

                iou_eval.update(masks_gt, masks_pred, batch['index_name'])
                valid_pbar.update(1)
                str_step_info = "Epoch: {epoch}/{epochs:4}.".format(
                    epoch=epoch, epochs=max_epoch_num
                )
                valid_pbar.set_postfix_str(str_step_info)
                if val_step % 40 == 0 or val_step == 1:
                    logging.info(
                        '#VAL#:{} Epoch [{:03d}/{:03d}], Step [{:04d}/200],  '.
                        format(datetime.now(), epoch, max_epoch_num, val_step))

            miou = iou_eval.compute()[0]['Mean Foreground IoU']
            logging.info(f'#VAL#:Epoch:{epoch}  miou:{miou}')
            iou_eval.reset()
            valid_pbar.clear()
            if miou > best_miou:
                torch.save(
                    model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
                    join(exp_path, "best_model.pth")
                )
                best_miou = miou
                print(f'Best mIoU has been updated to {best_miou:.2%}!')
                logging.info(f'Best mIoU has been updated to {best_miou:.2%}!')

args = parse()
import os


if __name__ == '__main__':
    

    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        used_gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        used_gpu = get_idle_gpu(gpu_num=1)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu[0])
    args.used_gpu, args.gpu_num = used_gpu, len(used_gpu)
    

    # launch the experiment process for both single-GPU and multi-GPU settings
    if len(args.used_gpu) == 1:
        main_worker(worker_id=0, worker_args=args)
    else:
        # initialize multiprocessing start method
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            try:
                mp.set_start_method('forkserver')
                print("Fail to initialize multiprocessing module by spawn method. "
                      "Use forkserver method instead. Please be careful about it.")
            except RuntimeError as e:
                raise RuntimeError(
                    "Your server supports neither spawn or forkserver method as multiprocessing start methods. "
                    f"The error details are: {e}"
                )

        # dist_url is fixed to localhost here, so only single-node DDP is supported now.
        args.dist_url = "tcp://127.0.0.1" + f':{get_idle_port()}'
        # spawn one subprocess for each GPU
        mp.spawn(main_worker, nprocs=args.gpu_num, args=(args,))
