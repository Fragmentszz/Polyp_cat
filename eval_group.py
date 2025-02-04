
from utils import test_save
from cat_sam.config import load_config
from cat_sam.build_model import build_model,build_dataloader_eval
import torch
import os
import logging  
if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # model_configs = ['cat-sam/cat_sam/config/model/6_1.yaml','cat-sam/cat_sam/config/model/6_3.yaml',
    #                  'cat-sam/cat_sam/config/model/6_7.yaml','cat-sam/cat_sam/config/model/7_3.yaml',
    #                  'cat-sam/cat_sam/config/model/8_2.yaml']
    model_configs = ['cat-sam/cat_sam/config/model/7_3.yaml',
                     'cat-sam/cat_sam/config/model/8_2.yaml']
    eval_dataset_config = '/applications/graduate_design/cat-sam/cat_sam/config/eval_dataset.yaml'
    test_datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    dataset_config = load_config(eval_dataset_config)
    dataloaders = build_dataloader_eval(dataset_config)
    root = '/remote-home/results'
    print(dataloaders)
    for model_config in model_configs:
        model_name = model_config.split('/')[-1].split('.')[0]
        config = load_config(model_config)
        model = build_model(config,device)
       
        save_root = f'{root}/{model_name}'
        if not os.path.exists(save_root):
                os.makedirs(save_root)
        logging.basicConfig(filename=os.path.join(save_root, 'eval.log'), 
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, 
                            filemode='a', 
                            datefmt='%Y-%m-%d %I:%M:%S %p')
        logging.info(
             f'=====Start Eval: model:{model_name}  dataset:{eval_dataset_config}  save_root:{save_root}'
        )
        for dl,dataset in zip(dataloaders,test_datasets):
            print(f'Testing on {dataset} dataset...')
            logging.info(f'Testing on {dataset} dataset...')
            save_path = f'{save_root}/{dataset}'
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            print(dl)
            dice,gd,iou = test_save(dl,model,device,save_path=save_path)
            logging.info(f'Mean val dice: {dice}')
            logging.info(f'Mean val gd: {gd}')
            logging.info(f'Mean val iou: {iou}')
