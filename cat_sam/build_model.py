from cat_sam.models.modeling import CATSAMT, CATSAMA,Reins
from cat_sam.models.segment_anything_ext import change_rein_cfg
from cat_sam.datasets.kvasir import KvasirDataset,KvasirDataset_test
import torch

def build_model(config,device,local_rank=None):
    model_type = config['model']['cat_type']
    if model_type == 'cat-t':
        model_class = CATSAMT
    elif model_type == 'cat-a':
        model_class = CATSAMA
    elif model_type == 'Reins':
        model_class = Reins
    else:
        raise ValueError(f"Model type {model_type} not supported")
    
    sam_type = config['model']['sam_type']
    if sam_type in ['rein_vit_l','rein__vit_h']:
        reins_config = config['model']['reins_config']
        reins_config = change_rein_cfg(model_type=sam_type,rein_cfg=reins_config)
    model = model_class(model_type=sam_type,rein_cfg=reins_config).to(device=device)
    ckpt_path = config['model']['ckpt_path']
    if ckpt_path is not None:
        model_state_dict = torch.load(ckpt_path, map_location=device)
        if 'model' in model_state_dict.keys():
            model_state_dict = model_state_dict['model']
        model.load_state_dict(model_state_dict)
    else:
        if torch.distributed.is_initialized():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
            )
    return model
from cat_sam.datasets.transforms import HorizontalFlip, VerticalFlip, RandomCrop
from torch.utils.data import DataLoader
from functools import partial
import random
import numpy as np
import os

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
def build_dataloader_train(config,max_object_num=None,local_rank=None):
    dataset_dir = config['dataset']['data_dir']
    dataset_class = config['dataset']['dataset']
    if dataset_class == 'polyp' or dataset_class == 'kvasir':
        dataset_class = KvasirDataset
    elif dataset_class == 'kvasir_test':
        dataset_class = KvasirDataset_test
    else:
        raise ValueError(f'invalid dataset name: {dataset_class}!')

    train_bs = config['dataset']['train_bs']
    val_bs = config['dataset']['val_bs']
    train_workers = config['dataset']['num_workers']
    val_workers = config['dataset']['num_workers']

    transforms = [VerticalFlip(p=0.5), HorizontalFlip(p=0.5), RandomCrop(scale=[0.1, 1.0], p=1.0)]
    train_dataset = dataset_class(
        data_dir=dataset_dir, train_flag=True, shot_num=config['dataset']['shot_num'],
        transforms=transforms, max_object_num=max_object_num
    )
    val_dataset = dataset_class(data_dir=dataset_dir, train_flag=False)


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
    return train_dataloader,val_dataloader

def build_dataloader_eval(config):
    dataset_dir = config['dataset']['data_dir']
    dataset_class = KvasirDataset_test
    dataset = config['dataset']['dataset']
    if dataset == 'divide':
        test_dl = []
        test_datasets = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
        for dataset in test_datasets:
            test_dataset = dataset_class(data_dir=os.path.join(dataset_dir,dataset))
            test_dataloader = DataLoader(
                dataset=test_dataset, shuffle=False, drop_last=False,
                batch_size=config['dataset']['batch_size'], num_workers=config['dataset']['num_workers'],
                collate_fn=test_dataset.collate_fn
            )
            test_dl.append(test_dataloader)
        return test_dl
    else:
        if dataset not in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']:
            raise ValueError(f'invalid dataset name: {dataset}!')
        test_dataset = dataset_class(data_dir=os.join(dataset_dir,dataset))
        test_dataloader = DataLoader(
            dataset=test_dataset, shuffle=False, drop_last=False,
            batch_size=config['dataset']['batch_size'], num_workers=config['dataset']['num_workers'],
            collate_fn=test_dataset.collate_fn
        )
        return test_dataloader
        
    

