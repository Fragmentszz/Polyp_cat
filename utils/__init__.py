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


if __name__ == '__main__':
    gt = np.zeros((512,512))
    res = np.zeros((512,512))

    gt[100:200,100:200] = 1
    res[150:250,150:250] = 1

    diff = get_dif(gt,res)

    diff.show()
