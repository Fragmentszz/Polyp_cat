import numpy as np
from PIL import Image
def get_dif(gt,res):
    res_1_gt_0 = np.logical_and(res, np.logical_not(gt))
    res_0_gt_1 = np.logical_and(np.logical_not(res), gt)
    res_1_gt_1 = np.logical_and(res, gt)
    # print((gt.shape[0], gt.shape[1], 3))
    diff = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    diff[res_1_gt_0] = [0, 255, 0]
    diff[res_0_gt_1] = [255, 0, 0]
    
    diff[res_1_gt_1] = [255, 255, 255]
    # print(res_0_gt_1.sum())
    # print(res_1_gt_0.sum())
    # print(diff)
    diff = Image.fromarray(diff)
    

    
    return diff

import torch.nn.functional as F
from monai.metrics import DiceMetric, MeanIoU, SurfaceDiceMetric, SSIMMetric, GeneralizedDiceScore
import argparse
import os
from os.path import join
import torch
from train import batch_to_cuda
from tqdm import tqdm
def test_save(test_dataloader,model,device,save_path=None):
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
                
                
                if save_path is not None:
                    res = res.squeeze().cpu().numpy()
                    res = np.round(res * 255).astype(np.uint8)
                    gt = gt.squeeze().cpu().numpy()
                    gt = np.round(gt * 255).astype(np.uint8)
                    diff = get_dif(gt,res)
                    
                    diff.save(os.path.join(save_path, str(name)+".png"))
                    name += 1
        return sum(batch_dice) / len(batch_dice),sum(batch_gd) / len(batch_gd),sum(batch_iou) / len(batch_iou)


if __name__ == '__main__':
    gt = np.zeros((512,512))
    res = np.zeros((512,512))

    gt[100:200,100:200] = 1
    res[150:250,150:250] = 1

    diff = get_dif(gt,res)

    diff.show()
