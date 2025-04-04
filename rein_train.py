import argparse
from genericpath import exists
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
        '--data_dir', default='./data', type=str,
        help="The directory that the datasets are placed. Default to be ./data"
    )
    parser.add_argument(
        '--num_workers', default=None, type=int,
        help="The num_workers argument used for the training and validation dataloaders. "
             "Default to be 1 for one-shot and 4 for 16- and full-shot."
    )
    parser.add_argument(
        '--train_bs', default=None, type=int,
        help="The batch size for the training dataloader. Default to be 1 for one-shot and 4 for 16- and full-shot."
    )
    parser.add_argument(
        '--val_bs', default=None, type=int,
        help="The batch size for the validation dataloader. Default to be 1 for one-shot and 4 for 16- and full-shot."
    )
    parser.add_argument(
        '--dataset', required=True, type=str, choices=['whu', 'sbu', 'kvasir','polyp'],
        help="Your target dataset. This argument is required."
    )
    parser.add_argument(
        '--shot_num', default=None, type=int, choices=[1, 16],
        help="The number of your target setting. For one-shot please give --shot_num 1. "
             "For 16-shot please give --shot_num 16. For full-shot please leave it blank. "
             "Default to be full-shot."
    )
    parser.add_argument(
        '--sam_type', default='vit_l', type=str, choices=['vit_b', 'vit_l', 'vit_h','rein_vit_l','rein_vit_h'],
        help='The type of the backbone SAM model. Default to be vit_l.'
    )
    parser.add_argument(
        '--cat_type', required=True, type=str, choices=['cat-a', 'cat-t','reins'],
        help='The type of the CAT-SAM model. This argument is required.'
    )
    
    cat_ckpt_dir = '/applications/graduate_design/model/finetuned/cat_sam/'
    CAT_ckpt_path_dict = dict(
        cat_t=dict(
            vit_b=f'{cat_ckpt_dir}/_vit_b_cat-t_16shot/best_model.pth',
            vit_l=f'{cat_ckpt_dir}/_vit_l_cat-t_16shot/best_model.pth',
            vit_h=f'{cat_ckpt_dir}/_vit_h_cat-t_16shot/best_model.pth',
        ),
        cat_a=dict(
            vit_b=f'{cat_ckpt_dir}/_vit_b_cat-a_16shot/best_model.pth',
            vit_l=f'{cat_ckpt_dir}/_vit_l_cat-a_16shot/best_model.pth',
            vit_h=f'{cat_ckpt_dir}/_vit_h_cat-a_16shot/best_model.pth',
        )
    )
    parser.add_argument(
        '--ckpt_path', required=False, type=str, default=None,
        help="The absolute path to your target checkpoint file. This argument is required."
    )
    parser.add_argument(
        '--rein_type', required=True, type=str, default=None,
        help="the type of rein"
    )
    parser.add_argument(
        '--if_evp_feature', required=False, type=str, default='True',choices=['True','False'],
        help='if the evp_feature is added into tokens'
    )
    parser.add_argument(
        '--local_block', required=False, type=str, default='False',choices=['True','False'],
        help='if the locak_attention block need to be reined'
    )
    parser.add_argument(
        '--connect_hq_token', required=False, type=str, default='True',choices=['True','False'],
        help='if B concat with hq_token'
    )
    return parser.parse_args()


def batch_to_cuda(batch, device):
    for key in batch.keys():
        if key in ['images', 'gt_masks', 'point_coords', 'box_coords', 'noisy_object_masks', 'object_masks']:
            batch[key] = [
                item.to(device=device, dtype=torch.float32) if item is not None else None for item in batch[key]
            ]
        elif key in ['point_labels']:
            batch[key] = [
                item.to(device=device, dtype=torch.long) if item is not None else None for item in batch[key]
            ]
    return batch


cat_ckpt_dir = '/applications/graduate_design/model/finetuned/cat_sam/'

CAT_ckpt_path_dict = dict(
    cat_t=dict(
        vit_b=f'{cat_ckpt_dir}/_vit_b_cat-t_16shot/best_model.pth',
        vit_l=f'{cat_ckpt_dir}/_vit_l_cat-t_16shot/best_model.pth',
        vit_h=f'{cat_ckpt_dir}/_vit_h_cat-t_16shot/best_model.pth',
    ),
    cat_a=dict(
        vit_b=f'{cat_ckpt_dir}/_vit_b_cat-a_16shot/best_model.pth',
        vit_l=f'{cat_ckpt_dir}/_vit_l_cat-a_16shot/best_model.pth',
        vit_h=f'{cat_ckpt_dir}/_vit_h_cat-a_16shot/best_model.pth',
    )
)
def main_worker(worker_id, worker_args):
    set_randomness()
    if isinstance(worker_id, str):
        worker_id = int(worker_id)
    
    

    gpu_num = len(worker_args.used_gpu)
    world_size = os.environ['WORLD_SIZE'] if 'WORLD_SIZE' in os.environ.keys() else gpu_num
    base_rank = os.environ['RANK'] if 'RANK' in os.environ.keys() else 0
    local_rank = base_rank * gpu_num + worker_id
    if gpu_num > 1:
        dist.init_process_group(backend='nccl', init_method=worker_args.dist_url,
                                world_size=world_size, rank=local_rank)

    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(device)

    if worker_args.cat_type == 'cat-t' and worker_args.dataset in ['kvasir', 'sbu']:
        transforms = [VerticalFlip(p=0.5), HorizontalFlip(p=0.5)]
    else:
        transforms = [VerticalFlip(p=0.5), HorizontalFlip(p=0.5), RandomCrop(scale=[0.1, 1.0], p=1.0)]

    max_object_num = None
    if worker_args.dataset == 'whu':
        dataset_class = WHUDataset
        max_object_num = 25
    elif worker_args.dataset == 'kvasir':
        dataset_class = KvasirDataset
    elif worker_args.dataset == 'sbu':
        dataset_class = SBUDataset
    elif worker_args.dataset == 'polyp':
        dataset_class = KvasirDataset
        worker_args.dataset = ''
    else:
        raise ValueError(f'invalid dataset name: {worker_args.dataset}!')

    dataset_dir = join(worker_args.data_dir, worker_args.dataset)
    
    train_dataset = dataset_class(
        data_dir=dataset_dir, train_flag=True, shot_num=worker_args.shot_num,
        transforms=transforms, max_object_num=max_object_num
    )
    val_dataset = dataset_class(data_dir=dataset_dir, train_flag=False)

    train_bs = worker_args.train_bs if worker_args.train_bs else (1 if worker_args.shot_num == 1 else 1)
    val_bs = worker_args.val_bs if worker_args.val_bs else 2
    train_workers, val_workers = 1 if worker_args.shot_num == 1 else 4, 2
    if worker_args.num_workers is not None:
        train_workers, val_workers = worker_args.num_workers, worker_args.num_workers

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_bs = int(train_bs / torch.distributed.get_world_size())
    print(f"Worker {local_rank} has been initialized successfully!")
    print(f"Train batch size: {train_bs}, Validation batch size: {val_bs}")
    print(f"Shuffle: {sampler is None}, Train workers: {train_workers}, Validation workers: {val_workers}")
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=train_bs, shuffle=sampler is None, num_workers=train_workers,
        sampler=sampler, drop_last=False, collate_fn=train_dataset.collate_fn,
        worker_init_fn=partial(worker_init_fn, base_seed=3407)
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=val_workers,
        drop_last=False, collate_fn=val_dataset.collate_fn
    )
    
    if worker_args.cat_type == 'cat-t':
        model_class = CATSAMT
    elif worker_args.cat_type == 'cat-a':
        model_class = CATSAMA
    elif worker_args.cat_type == 'reins':
        model_class = Reins
    else:
        raise ValueError(f'invalid cat_type: {worker_args.cat_type}!')

    
    model = None
    reins_config = None
    if not os.path.exists(worker_args.exp_dir):
        os.mkdir(worker_args.exp_dir)
    exp_path = join(
        worker_args.exp_dir,
        f'{worker_args.rein_type}_{worker_args.dataset}_{worker_args.sam_type}_{worker_args.cat_type}_{worker_args.shot_num if worker_args.shot_num else "full"}shot'
    )
    print("=====================eo==========================")
    print(exp_path)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    logging.basicConfig(filename=os.path.join(exp_path,'log.log'), format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(f"====Train: {args.cat_type}_{args.sam_type}_{args.shot_num if args.shot_num else 'full'}shot ====")
    logging.info("Config")
    logging.info(f'Args:{args}')
    if worker_args.sam_type in ['rein_vit_l','rein__vit_h']:
        reins_config=dict(
            token_length=128,
            # link_token_to_query=True,
            # lora_dim=16,
            zero_mlp_delta_f=False,  # v2
        )
        if worker_args.rein_type == 'LoRAReins':
            reins_config['lora_dim'] = 16
        if worker_args.cat_type == 'reins':
            reins_config['link_token_to_query'] = True
        reins_config['if_evp_feature'] = worker_args.if_evp_feature
        reins_config['local_block'] = worker_args.local_block
        reins_config['connect_hq_token'] = worker_args.connect_hq_token
        
        reins_config['rein_type'] = worker_args.rein_type
        from cat_sam.models.segment_anything_ext import change_rein_cfg
        reins_config = change_rein_cfg(model_type=worker_args.sam_type,rein_cfg=reins_config)
        logging.info(f"Reins config:{reins_config}")
        print(f"Reins config:{reins_config}") 
    # model = model_class(model_type=worker_args.sam_type,rein_cfg=reins_config).to(device=device)
    # if torch.distributed.is_initialized():
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    #     )
    if worker_args.ckpt_path is None:
        model = model_class(model_type=worker_args.sam_type,rein_cfg=reins_config).to(device=device)
        if torch.distributed.is_initialized():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )
    else:
        model = model_class(model_type=worker_args.sam_type,rein_cfg=reins_config).to(device=device)
        model_state_dict = torch.load(worker_args.ckpt_path, map_location=device)
        if 'model' in model_state_dict.keys():
            model_state_dict = model_state_dict['model']
        model.load_state_dict(model_state_dict)
    model.train()



    optimizer = torch.optim.AdamW(
        params=[p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-2
    )
    # full-shot
    if worker_args.shot_num is None:
        max_epoch_num, valid_per_epochs = 30, 1
    # one-shot
    elif worker_args.shot_num == 1:
        max_epoch_num, valid_per_epochs = 2000, 20
    # 16-shot
    elif worker_args.shot_num == 16:
        max_epoch_num, valid_per_epochs = 200, 2
    else:
        raise RuntimeError
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=max_epoch_num, eta_min=1e-5
    )

    best_miou = 0
    if worker_args.dataset == 'hqseg44k':
        iou_eval = SamHQIoU()
    else:
        if worker_args.dataset in ['jsrt', 'fls']:
            class_names = train_dataset.class_names
        else:
            class_names = ['Background', 'Foreground']
        iou_eval = StreamSegMetrics(class_names=class_names)

    
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
