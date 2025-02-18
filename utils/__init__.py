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



def get_res(gt,res,img):
    img = img.detach()
    img = img.permute(1,2,0).cpu().numpy()
    img = np.array(img,dtype=np.uint8)
    # print(res.shape,img.shape)
    res_1_gt_0 = np.logical_and(res, np.logical_not(gt))
    res_0_gt_1 = np.logical_and(np.logical_not(res), gt)
    res_1_gt_1 = np.logical_and(res, gt)
    # print((gt.shape[0], gt.shape[1], 3))
    diff = img
    # diff = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    # diff[res_1_gt_0] = [0, 255, 0]
    # # diff[res_0_gt_1] = [255, 0, 0]
    # print(res==255)
    diff[res==255] = [0,255,0]
    diff[gt==255] = [255,0,0]
    
    diff[res_1_gt_1==255] = [255, 255, 255]
    # diff[res==0] = [255,0,0]
    
    # diff[res_1_gt_1] = [255, 255, 255]
    # # print(res_0_gt_1.sum())
    # # print(res_1_gt_0.sum())
    # # print(diff)
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
def test_save(test_dataloader,model,device,save_path=None,save_func=get_dif):
    if save_path is not None and not os.path.exists(save_path):
        os.mkdir(save_path)
    
    with torch.no_grad():
        batch_dice = []
        batch_gd = []
        batch_iou = []
        for test_step, batch in enumerate(tqdm(test_dataloader)):
            batch = batch_to_cuda(batch, device)
            imgs = batch['images']
            model.set_infer_img(img=imgs)

            masks_pred = model.infer(box_coords=batch['box_coords'])
            masks_gt = batch['gt_masks']
            names = batch['index_name']
            
            
            for mask_pred, mask_gt,name,img in zip(masks_pred, masks_gt,names,imgs):
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
                    if save_func == get_res:
                        diff = save_func(gt,res,img)
                    else:
                        diff = save_func(gt,res)
                    
                    diff.save(os.path.join(save_path, str(name)+".png"))
        return sum(batch_dice) / len(batch_dice),sum(batch_gd) / len(batch_gd),sum(batch_iou) / len(batch_iou)

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


def test_token(test_dataloader,model,device,save_path=None,save_func=get_res):
    
    A_ori = model.image_encoder.reins.A.detach().clone()
   
    m = A_ori.shape[1]
    dim = A_ori.shape[2]
    pre = None
    print(m,dim)
    for test_token_id in range(m):
        print(test_token_id)
        
        cp = A_ori[0,test_token_id].clone()
        zeros = torch.zeros_like(A_ori)
        zeros[0,test_token_id] = cp
        model.image_encoder.reins.A = torch.nn.Parameter(zeros)
        B_ori = model.image_encoder.reins.B[0].detach().clone()
        # print(model.image_encoder.reins.A)
        tmp = model.image_encoder.reins.A[0] @ B_ori
        if pre is not None:
            print(torch.abs(tmp-pre).sum())
            assert not torch.allclose(tmp,pre),"不太对劲"
            
        pre = tmp
        non_zero_rows = torch.nonzero(tmp.sum(dim=1) / 128 /256)
        print(test_token_id,non_zero_rows)
        assert test_token_id == non_zero_rows, "不太对劲"
        
        if save_path is not None:
            now_save_path = os.path.join(save_path,str(test_token_id))
        else:
            now_save_path = None
        test_save(test_dataloader,model,device,now_save_path,save_func)
            

if __name__ == '__main__':
    gt = np.zeros((512,512))
    res = np.zeros((512,512))

    gt[100:200,100:200] = 1
    res[150:250,150:250] = 1

    diff = get_dif(gt,res)

    diff.show()
