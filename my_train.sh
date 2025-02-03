expected_hostname="ins-"


current_hostname=$(hostname)

if echo "$current_hostname" | grep -q "$expected_hostname"; then
    python rein_train.py --exp_dir=/applications/graduate_design/model/finetuned/cat_sam --data_dir=/remote-home/ --num_workers=4 --sam_type=rein_vit_l --cat_type=cat-a --dataset=polyp --shot_num=16 --train_bs=1 --val_bs=1 --rein_type=Reins_Attention7 --if_evp_feature=False --local_block=True
else
    python rein_train.py --exp_dir=/applications/graduate_design/model/finetuned/cat_sam --data_dir=/applications/graduate_design/ --num_workers=4 --sam_type=rein_vit_l --cat_type=cat-a --dataset=polyp --shot_num=16 --rein_type=Reins_Attention7 --if_evp_feature=False --local_block=True
fi