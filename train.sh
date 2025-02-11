expected_hostname="ins-18857555890"


current_hostname=$(hostname)


if echo "$current_hostname" | grep -q "$expected_hostname"; then
    python train.py --exp_dir=/applications/graduate_design/model/finetuned/cat_sam --data_dir=/root/autodl-tmp --num_workers=4 --sam_type=vit_l --cat_type=cat-a --dataset=polyp --shot_num=16 --train_bs=1 --val_bs=16 --ckpt_path=/applications/graduate_design/model/finetuned/cat_sam/_vit_l_cat-a_16shot/best_model.pth
else
    python train.py --exp_dir=/applications/graduate_design/model/finetuned/cat_sam --data_dir=/applications/graduate_design/ --num_workers=4 --sam_type=vit_l --cat_type=cat-a --dataset=polyp --shot_num=16
fi