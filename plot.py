
from utils import test_save
from cat_sam.config import load_config
from cat_sam.build_model import build_model,build_dataloader_eval
import torch
import os
import logging  
if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_configs = ['/applications/graduate_design/cat-sam/cat_sam/config/model/7_7___.yaml']
    eval_dataset_config = '/applications/graduate_design/cat-sam/cat_sam/config/plot_dataset.yaml'
    test_datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    dataset_config = load_config(eval_dataset_config)
    dataloaders = build_dataloader_eval(dataset_config)
    root = '/root/autodl-fs/results'
    print(dataloaders)
    for model_config in model_configs:
        model_name = model_config.split('/')[-1].split('.')[0]
        config = load_config(model_config)
        model = build_model(config,device)
        
        save_root = f'{root}/{model_name}'
        if not os.path.exists(save_root):
                os.makedirs(save_root)
        save_path = f'{save_root}/{dataset_config['dataset']['dataset']}'
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        dice,gd,iou = test_save(dataloaders,model,device,save_path=save_path)
            

