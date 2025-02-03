
from myeval import test_save
from cat_sam.config import load_config
from cat_sam.build_model import build_model,build_dataloader_eval

if __name__ == '__main__':
    model_configs = ['cat-sam/cat_sam/config/model/6_1.yaml','cat-sam/cat_sam/config/model/6_3.yaml',
                     'cat-sam/cat_sam/config/model/6_7.yaml','cat-sam/cat_sam/config/model/7_3.yaml'
                     'cat-sam/cat_sam/config/model/8_2.yaml']
    for model_config in model_configs:
        config = load_config(model_config)
        model = build_model(config)
        dataloader = build_dataloader_eval(config)
        test_save(dataloader,model,save_path='cat-sam/cat_sam/eval_results')