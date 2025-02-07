import torch
from torch import nn

from cat_sam.models.module_lib import Adapter, PromptGenerator
from .encoders import CATSAMAImageEncoder,CATSAMTImageEncoder,SAMImageEncodeWrapper

from .reins import Reins,LoRAReins,Reins_Attention,My_LoRAReins,Reins_Attention2,Reins_Attention2_upd,Reins_Attention2_upd2,Reins_Attention3

from .reins import Reins_Attention3_v2
from .reins2 import EVP,Reins_Attention4,Reins_Attention5, Reins_Attention6, Reins_Attention7, Reins_Attention8
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
    'Reins_Attention5':Reins_Attention5,
    'Reins_Attention6':Reins_Attention6,
    'Reins_Attention7':Reins_Attention7,
    'Reins_Attention8':Reins_Attention8
}

class ReinsImageEncoder(SAMImageEncodeWrapper):

    def __init__(
            self, ori_sam, hq_token: torch.Tensor, reins_cfg = None
    ):
        super(ReinsImageEncoder, self).__init__(ori_sam=ori_sam, fix=True)

        self.rein_enabled_layers = self.sam_img_encoder.global_attn_indexes
        self.rein_cfg = reins_cfg

        rein_cls = cls_dic[reins_cfg['type']]
        reins_cfg.pop('type')
        self.reins =  rein_cls(**self.rein_cfg) if self.rein_cfg is not None else None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sam_img_encoder.patch_embed(x)

      
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
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
        self.rein_enabled_layers = self.sam_img_encoder.global_attn_indexes
        reins_cfg['num_layers'] = len(self.rein_enabled_layers)
        reins_cfg['embed_dims_ratio'] = 0.25
        reins_cfg['hq_token'] = hq_token
        self.EVP = EVP(img_size=self.sam_img_encoder.img_size,patch_size=self.sam_img_encoder.patch_embed.proj.kernel_size[0],
                        embed_dim=reins_cfg['embed_dims'],freq_nums=0.25)
        rein_cls = cls_dic[reins_cfg['type']]
        print(reins_cfg)
        reins_cfg.pop('type')
        
        

        self.reins = rein_cls(**self.rein_cfg) if self.rein_cfg is not None else None
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.sam_img_encoder.patch_embed(x)
        

        evp = self.EVP(inp)
        
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed
        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = blk(x)
            B, H, W, C = x.shape
            
            if self.reins is not None and i in self.rein_enabled_layers :
                x = self.reins.forward(
                    x.view(B, -1, C),
                    self.rein_enabled_layers.index(i),
                    batch_first=True,
                    has_cls_token=False,
                    evp_feature=evp
                ).view(B, H, W, C)
            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings


# 4+1 layers
class MyCATSAMAImageEncoder4(CATSAMAImageEncoder):
    def __init__(
            self, ori_sam, hq_token: torch.Tensor,reins_cfg=None
    ):
        super(MyCATSAMAImageEncoder4, self).__init__(ori_sam=ori_sam,hq_token=hq_token)

        self.rein_enabled_layers = self.sam_img_encoder.global_attn_indexes
        reins_cfg['num_layers'] = len(self.rein_enabled_layers)
        self.reins_num_layers = len(self.rein_enabled_layers)
        reins_cfg['embed_dims_ratio'] = 0.25
        reins_cfg['hq_token'] = hq_token
        if type(reins_cfg['if_evp_feature']) == str:
            self.if_evp_feature = reins_cfg['if_evp_feature'] == 'True'
        else:
            self.if_evp_feature = reins_cfg['if_evp_feature']
        if type(reins_cfg['local_block']) == str:
            self.if_local_block = reins_cfg['local_block'] == 'True'
        else:
            self.if_local_block = reins_cfg['local_block']
        
        self.EVP2 = EVP(img_size=self.sam_img_encoder.img_size,patch_size=self.sam_img_encoder.patch_embed.proj.kernel_size[0],
                        embed_dim=reins_cfg['embed_dims'],freq_nums=0.25)
        self.EVP_f = nn.Linear(self.EVP2.patch_embed.num_patches,reins_cfg['token_length'])
        rein_cls = cls_dic[reins_cfg['rein_type']]
        self.hq_token = hq_token
        
        print("==============look:",'connect_hq_token' in reins_cfg)
        required_keys = ['embed_dims','num_layers','patch_size','token_length','embed_dims_ratio','embed_dims_ratio','hq_token','scale_init','zero_mlp_delta_f',
                         'connect_hq_token']
        self.rein_cfg = {}

        for key in required_keys:
            if key in reins_cfg:
                self.rein_cfg[key] = reins_cfg[key]
        print(self.rein_cfg.keys())
        self.reins = rein_cls(**self.rein_cfg) if self.rein_cfg is not None else None
        
    def get_hq_token(self):
        return self.reins.get_hq_token()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.sam_img_encoder.patch_embed(x)
        if self.if_evp_feature:
            evp_feature = self.EVP2(inp)
            fB, fC, fH,fW = evp_feature.shape
            # m*c
            evp_feature = evp_feature.reshape(fB,fC,-1)
            evp_feature = self.EVP_f(evp_feature).permute(0,2,1)
        else:
            evp_feature = None
        
        if self.sam_img_encoder.pos_embed is not None:
            x = x + self.sam_img_encoder.pos_embed

        interm_embeddings = []
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        for i, blk in enumerate(self.sam_img_encoder.blocks):
            x = blk(x)
            B, H, W, C = x.shape
            
            if self.reins is not None :
                if i in self.rein_enabled_layers :
                    x = self.reins.forward(
                        x.view(B, -1, C),
                        self.rein_enabled_layers.index(i),
                        batch_first=True,
                        has_cls_token=False,
                        evp_feature= evp_feature
                    ).view(B, H, W, C)
                elif self.if_local_block:
                    x = self.reins.forward(
                        x.view(B, -1, C),
                        self.reins_num_layers,
                        batch_first=True,
                        has_cls_token=False,
                        evp_feature=evp_feature
                    ).view(B, H, W, C)

            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.sam_img_encoder.neck(x.permute(0, 3, 1, 2))
        return x, interm_embeddings