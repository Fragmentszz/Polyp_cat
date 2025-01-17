import torch
from torch import nn

from cat_sam.models.module_lib import Adapter, PromptGenerator
from .encoders import CATSAMAImageEncoder,CATSAMTImageEncoder

from .reins import Reins,LoRAReins,My_LoRAReins
class ReinCATSAMTImageEncoder(CATSAMTImageEncoder):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,reins_cfg=None
    ):
        super(ReinCATSAMTImageEncoder, self).__init__(ori_sam=ori_sam,hq_token=hq_token)
        self.rein_enabled_layers = self.sam_img_encoder.global_attn_indexes

        self.rein_cfg = reins_cfg

        print(self.rein_cfg)
        

        # self.reins:Reins =  LoRAReins(**self.rein_cfg) if self.rein_cfg is not None else None
        self.reins:Reins =  My_LoRAReins(**self.rein_cfg) if self.rein_cfg is not None else None

    def forward(self, x):
        x = self.sam_img_encoder.patch_embed(x)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed

        hq_prompt_tokens = []
        for i in range(0, len(self.hq_token_proj)):
            hq_prompt_tokens.append(self.hq_token_proj[i](self.hq_token).unsqueeze(0))

        interm_embeddings = []
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = blk(x, hq_prompt_tokens[i])
            B, H, W, C = x.shape
            if self.reins is not None and i in self.rein_enabled_layers :
                x = self.reins.forward(
                    x.view(B, -1, C),
                    self.rein_enabled_layers.index(i),
                    batch_first=True,
                    has_cls_token=False,
                ).view(B, H, W, C)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings






class ReinCATSAMAImageEncoder(CATSAMAImageEncoder):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,reins_cfg=None
    ):
        super(ReinCATSAMAImageEncoder, self).__init__(ori_sam=ori_sam,hq_token=hq_token)

        self.rein_enabled_layers = self.sam_img_encoder.global_attn_indexes

        self.rein_cfg = reins_cfg

        print(self.rein_cfg)
        
        # self.rein_cfg['num_layers'] = len(self.rein_enabled_layers)
        # self.rein_cfg.remove

        # self.reins:Reins =  LoRAReins(**self.rein_cfg) if self.rein_cfg is not None else None
        self.reins:Reins =  My_LoRAReins(**self.rein_cfg) if self.rein_cfg is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.sam_img_encoder.patch_embed(x)

        embedding_feature = self.prompt_generator.init_embeddings(x)
        handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
        hq_feature = torch.cat(
            [self.shared_up_proj(down_proj(self.hq_token)).unsqueeze(-1) for down_proj in self.hq_token_down_proj],
            dim=-1
        )
        prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature, hq_feature=hq_feature)
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed

        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = prompt[i].reshape(B, H, W, -1) + x
            x = blk(x)
            B, H, W, C = x.shape
            if self.reins is not None and i in self.rein_enabled_layers :
                x = self.reins.forward(
                    x.view(B, -1, C),
                    self.rein_enabled_layers.index(i),
                    batch_first=True,
                    has_cls_token=False,
                ).view(B, H, W, C)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings
