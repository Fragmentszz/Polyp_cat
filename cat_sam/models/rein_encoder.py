import torch
from torch import nn

from cat_sam.models.module_lib import Adapter, PromptGenerator
from .encoders import CATSAMAImageEncoder,CATSAMTImageEncoder

from .reins import Reins,LoRAReins,Reins_Attention,My_LoRAReins,Reins_Attention2,Reins_Attention2_upd,Reins_Attention2_upd2,Reins_Attention3

from .reins import Reins_Attention3_v2
from .reins2 import EVP,Reins_Attention4,Reins_Attention5
cls_dic = {
    'Reins':Reins,
    'LoRAReins':LoRAReins,
    'Reins_Attention':Reins_Attention,
    'My_LoRAReins':Reins_Attention2,
    'Reins_Attention2':Reins_Attention2,
    'Reins_Attention2_upd':Reins_Attention2_upd,
    'Reins_Attention2_upd2':Reins_Attention2_upd2,
    'Reins_Attention3':Reins_Attention3,
    'Reins_Attention3_v2':Reins_Attention3_v2,
    'Reins_Attention4':Reins_Attention4,
    'Reins_Attention5':Reins_Attention5
}


class ReinCATSAMTImageEncoder(CATSAMTImageEncoder):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,reins_cfg=None
    ):
        super(ReinCATSAMTImageEncoder, self).__init__(ori_sam=ori_sam,hq_token=hq_token)
        self.rein_enabled_layers = self.sam_img_encoder.global_attn_indexes

        self.rein_cfg = reins_cfg

        print(self.rein_cfg)
        

        self.reins:Reins =  LoRAReins(**self.rein_cfg) if self.rein_cfg is not None else None

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

        self.reins:Reins =  LoRAReins(**self.rein_cfg) if self.rein_cfg is not None else None

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


class MyCATSAMAImageEncoder(CATSAMAImageEncoder):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,reins_cfg=None
    ):
        super(MyCATSAMAImageEncoder, self).__init__(ori_sam=ori_sam,hq_token=hq_token)

        self.rein_enabled_layers = self.sam_img_encoder.global_attn_indexes

        self.rein_cfg = reins_cfg
        reins_cfg['num_layers'] = len(self.sam_img_encoder.blocks)
        reins_cfg['embed_dims_ratio'] = 32
        reins_cfg['hq_token'] = hq_token
        
        self.hq_token = hq_token
        patch_height = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[0]
        patch_width = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[1]
        

        self.reins = Reins_Attention(**self.rein_cfg) if self.rein_cfg is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.sam_img_encoder.patch_embed(x)

        # embedding_feature = self.prompt_generator.init_embeddings(x)
        # handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
        # hq_feature = torch.cat(
        #     [self.shared_up_proj(down_proj(self.hq_token)).unsqueeze(-1) for down_proj in self.hq_token_down_proj],
        #     dim=-1
        # )
        # prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature, hq_feature=hq_feature)
        # if self.sam_img_encoder.pos_embed is not None:
        #     x = x + self.sam_img_encoder.pos_embed

        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            # x = prompt[i].reshape(B, H, W, -1) + x
            x = blk(x)
            B, H, W, C = x.shape
            if self.reins is not None:
                x = self.reins.forward(
                    x.view(B, -1, C),
                    i,
                    batch_first=True,
                    has_cls_token=False,
                ).view(B, H, W, C)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings

class MyCATSAMAImageEncoder2(CATSAMAImageEncoder):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,reins_cfg=None
    ):
        super(MyCATSAMAImageEncoder2, self).__init__(ori_sam=ori_sam,hq_token=hq_token)

        self.rein_enabled_layers = self.sam_img_encoder.global_attn_indexes

        self.rein_cfg = reins_cfg
        reins_cfg['num_layers'] = len(self.sam_img_encoder.blocks)
        reins_cfg['embed_dims_ratio'] = 0.25
        reins_cfg['hq_token'] = hq_token
        # self.EVP = EVP(img_size=self.sam_img_encoder.img_size,patch_size=self.sam_img_encoder.patch_embed.proj.kernel_size[0],
        #                 embed_dim=reins_cfg['token_length'],freq_nums=0.25)
        # reins_cfg['EVP_size'] = self.EVP.patch_embed.num_patches


        rein_cls = cls_dic[reins_cfg['type']]



        reins_cfg.pop('type')
        self.hq_token = hq_token
        print(reins_cfg)
        patch_height = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[0]
        patch_width = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[1]
        

        self.reins = rein_cls(**self.rein_cfg) if self.rein_cfg is not None else None
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.sam_img_encoder.patch_embed(x)

        evp = self.EVP(inp)

        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            # x = prompt[i].reshape(B, H, W, -1) + x
            x = blk(x)
            B, H, W, C = x.shape
            if self.reins is not None:
                x = self.reins.forward(
                    x.view(B, -1, C),
                    i,
                    batch_first=True,
                    has_cls_token=False,
                ).view(B, H, W, C)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings
    
class MyCATSAMAImageEncoder3(CATSAMAImageEncoder):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,reins_cfg=None
    ):
        super(MyCATSAMAImageEncoder3, self).__init__(ori_sam=ori_sam,hq_token=hq_token)

        self.rein_cfg = reins_cfg
        reins_cfg['num_layers'] = len(self.sam_img_encoder.blocks)
        reins_cfg['embed_dims_ratio'] = 0.25
        reins_cfg['hq_token'] = hq_token
        self.EVP = EVP(img_size=self.sam_img_encoder.img_size,patch_size=self.sam_img_encoder.patch_embed.proj.kernel_size[0],
                        embed_dim=reins_cfg['embed_dims'],freq_nums=0.25)
        # reins_cfg['EVP_size'] = self.EVP.patch_embed.num_patches


        rein_cls = cls_dic[reins_cfg['type']]



        reins_cfg.pop('type')
        self.hq_token = hq_token
        print(reins_cfg)
        patch_height = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[0]
        patch_width = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[1]
        

        self.reins = rein_cls(**self.rein_cfg) if self.rein_cfg is not None else None
        
    def get_hq_token(self):
        return self.reins.get_hq_token()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.sam_img_encoder.patch_embed(x)

        evp = self.EVP(inp)
        evp_feature = self.EVP2(inp)
        

        

        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = blk(x)
            B, H, W, C = x.shape
            
            if self.reins is not None:
                x = self.reins.forward(
                    x.view(B, -1, C),
                    i,
                    batch_first=True,
                    has_cls_token=False,
                    evp_feature=evp
                ).view(B, H, W, C)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings

class MyCATSAMAImageEncoder4(CATSAMAImageEncoder):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor,reins_cfg=None
    ):
        super(MyCATSAMAImageEncoder4, self).__init__(ori_sam=ori_sam,hq_token=hq_token)

        self.rein_cfg = reins_cfg
        reins_cfg['num_layers'] = len(self.sam_img_encoder.blocks)
        reins_cfg['embed_dims_ratio'] = 0.25
        reins_cfg['hq_token'] = hq_token
        patch_size = self.sam_img_encoder.img_size // 32
        
        self.EVP2 = EVP(img_size=self.sam_img_encoder.img_size,patch_size=patch_size,
                        embed_dim=reins_cfg['token_length'],freq_nums=0.25)
        # reins_cfg['EVP_size'] = self.EVP.patch_embed.num_patches


        rein_cls = cls_dic[reins_cfg['type']]



        reins_cfg.pop('type')
        self.hq_token = hq_token
        print(reins_cfg)
        patch_height = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[0]
        patch_width = self.sam_img_encoder.img_size / self.sam_img_encoder.patch_embed.proj.kernel_size[1]
        

        self.reins = rein_cls(**self.rein_cfg) if self.rein_cfg is not None else None
        
    def get_hq_token(self):
        return self.reins.get_hq_token()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.sam_img_encoder.patch_embed(x)

        # evp = self.EVP(inp)
        evp_feature = self.EVP2(inp)
        # print("evp_feature",evp_feature.shape)
        

        fB, fC, fH,fW = evp_feature.shape
        # m*c
        evp_feature = evp_feature.reshape(fB,fC,-1)

        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = blk(x)
            B, H, W, C = x.shape
            
            
            if self.reins is not None:
                x = self.reins.forward(
                    x.view(B, -1, C),
                    i,
                    batch_first=True,
                    has_cls_token=False,
                    evp_feature=evp_feature
                ).view(B, H, W, C)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings