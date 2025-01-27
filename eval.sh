expected_hostname="ins-"


current_hostname=$(hostname)


if echo "$current_hostname" | grep -q "$expected_hostname"; then
    python eval_group.py  --data_dir=/remote-home/dataset/polyp/test --num_workers=4 --sam_type=rein_vit_l --cat_type=cat-a --dataset=divide --batch_size=4 --ckpt_path=/applications/graduate_design/model/finetuned/cat_sam/_rein_vit_l_cat-a_16shot/best_model.pth --rein_type=Reins_Attention2_upd2
else
    python eval_group.py  --data_dir=/applications/graduate_design/dataset/polyp/test --num_workers=4 --sam_type=rein_vit_l --cat_type=cat-a --dataset=divide --batch_size=4 --ckpt_path=/applications/graduate_design/model/finetuned/cat_sam/_rein_vit_l_cat-a_16shot/best_model.pth --rein_type=Reins_Attention2_upd2
fi