expected_hostname="ins-18857555890-20250107-gpu-dbcloud-master-a0fe70e1-65797ns59q"


current_hostname=$(hostname)


if [ "$current_hostname" == "$expected_hostname" ]; then
    python eval.py  --data_dir=/remote-home/dataset/polyp/test --num_workers=4 --sam_type=rein_vit_l --cat_type=cat-a --dataset=divide --batch_size=4 --ckpt_path=/applications/graduate_design/model/finetuned/cat_sam/_rein_vit_l_cat-a_16shot/best_model.pth --rein_type=Reins_Attention2_upd2
else
    python eval.py  --data_dir=/applications/graduate_design/dataset/polyp/test --num_workers=4 --sam_type=rein_vit_l --cat_type=cat-a --dataset=divide --batch_size=4 --ckpt_path=/applications/graduate_design/model/finetuned/cat_sam/_rein_vit_l_cat-a_16shot/best_model.pth --rein_type=Reins_Attention2_upd2
fi