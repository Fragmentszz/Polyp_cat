expected_hostname="ins-"


current_hostname=$(hostname)

# if echo "$current_hostname" | grep -q "$expected_hostname"; then
#     python my_train.py --exp_dir=/applications/graduate_design/model/finetuned/cat_sam --model_config=/applications/graduate_design/cat-sam/cat_sam/config/model/7_7_.yaml --dataset_config=/applications/graduate_design/cat-sam/cat_sam/config/train_dataset.yaml
# else
#     python rein_train.py --exp_dir=/applications/graduate_design/model/finetuned/cat_sam --data_dir=/applications/graduate_design/ --num_workers=4 --sam_type=rein_vit_l --cat_type=cat-a --dataset=polyp --shot_num=16 --rein_type=Reins_Attention8 --if_evp_feature=False --local_block=True --connect_hq_token=False
# fi
python my_train.py --exp_dir=/applications/graduate_design/model/finetuned/cat_sam --model_config=/applications/graduate_design/cat-sam/cat_sam/config/model/7_7___24.yaml --dataset_config=/applications/graduate_design/cat-sam/cat_sam/config/train_dataset.yaml
