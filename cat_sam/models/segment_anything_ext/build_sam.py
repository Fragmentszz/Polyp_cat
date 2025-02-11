# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


parm_dic = {
    'rein_vit_b':{
        'embed_dims':768,
        'patch_size':16,
        'global_attn_indexes':[2, 5, 8, 11],
    },
    'rein_vit_l':{
        'embed_dims':1024,
        'patch_size':16,
        'global_attn_indexes':[5, 11, 17, 23],
    },
    'rein_vit_h':{
        'embed_dims':1280,
        'patch_size':16,
        'global_attn_indexes':[7, 15, 23, 31],
    },
    
}
def change_rein_cfg(model_type,rein_cfg):
    res = rein_cfg
    print(rein_cfg)
    res['embed_dims'] = parm_dic[model_type]['embed_dims']
    res['patch_size'] = parm_dic[model_type]['patch_size']
    # res['global_attn_indexes'] = parm_dic[model_type]['global_attn_indexes']
    res['num_layers'] = len(parm_dic[model_type]['global_attn_indexes'])
    return res
    
build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "rein_vit_l": build_sam_vit_l,
    "rein_vit_b": build_sam_vit_b,
    "rein_vit_h": build_sam_vit_h

}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


# def _build_sam_rein(
#     encoder_embed_dim,
#     encoder_depth,
#     encoder_num_heads,
#     encoder_global_attn_indexes,
#     checkpoint=None,

#     rein_cfg=None
# ):
#     prompt_embed_dim = 256
#     image_size = 1024
#     vit_patch_size = 16
#     image_embedding_size = image_size // vit_patch_size
#     sam = Sam(
#         image_encoder=ReinImageEncoderViT(
#             depth=encoder_depth,
#             embed_dim=encoder_embed_dim,
#             img_size=image_size,
#             mlp_ratio=4,
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             num_heads=encoder_num_heads,
#             patch_size=vit_patch_size,
#             qkv_bias=True,
#             use_rel_pos=True,
#             global_attn_indexes=encoder_global_attn_indexes,
#             window_size=14,
#             out_chans=prompt_embed_dim,
#             reins_cfg=rein_cfg
#         ),
#         prompt_encoder=PromptEncoder(
#             embed_dim=prompt_embed_dim,
#             image_embedding_size=(image_embedding_size, image_embedding_size),
#             input_image_size=(image_size, image_size),
#             mask_in_chans=16,
#         ),
#         mask_decoder=MaskDecoder(
#             num_multimask_outputs=3,
#             transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=prompt_embed_dim,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=prompt_embed_dim,
#             iou_head_depth=3,
#             iou_head_hidden_dim=256,
#         ),
#         pixel_mean=[123.675, 116.28, 103.53],
#         pixel_std=[58.395, 57.12, 57.375],
#     )
#     sam.eval()
#     if checkpoint is not None:
#         with open(checkpoint, "rb") as f:
#             state_dict = torch.load(f)
#         sam.load_state_dict(state_dict)
#     return sam