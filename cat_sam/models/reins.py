
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
from .module_lib import Adapter, PromptGenerator


class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            return feats, querys
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)
        return delta_f



class LoRAReins(Reins):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]


class My_LoRAReins(Reins):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([1, self.token_length, self.lora_dim])
        )



        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            print("========================layer=-1============================")
            ta = torch.concat([self.learnable_tokens_a]*self.num_layers,dim=0)
            return ta @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[0] @ self.learnable_tokens_b[layer]



class Reins_Attention(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=32,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.hq_token = hq_token
        self.use_softmax = use_softmax
        # self.hxw = patch_height * patch_width
        self.hq_token_down_proj = nn.Sequential(
            *[Adapter(in_features=hq_token.size(-1), mlp_ratio=0.125, add_last_layer=False)
              for _ in range(self.embed_dims // embed_dims_ratio)]
        )

        # assert patch_height is not None and patch_width is not None, "patch_height and patch_width should be provided"
        self.shared_up_proj = nn.Linear(
            in_features=int(hq_token.size(-1) * 0.125),
            out_features=int(embed_dims)
        )
        self.scale_init = scale_init
        self.create_model()


    def create_model(self):
        # self.learnable_tokens = nn.Parameter(
        #     torch.empty([self.num_layers, self.token_length, self.embed_dims])
        # )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.hq_token2token = nn.Linear(len(self.hq_token_down_proj), self.token_length)
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_() 
    def get_mlps(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.mlp_token2feat, self.mlp_delta_f
        else:
            return self.mlp_token2feat, self.mlp_delta_f
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        mlp_token2feat, mlp_delta_f = self.get_mlps(layers)

        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn,
            mlp_token2feat(tokens),
        )
        delta_f = mlp_delta_f(delta_f + feats)
        return delta_f
    def get_tokens(self, x:torch.Tensor,layer: int) -> Tensor:

        hq_feature = torch.cat(
            [self.shared_up_proj(down_proj(self.hq_token)).unsqueeze(-1) for down_proj in self.hq_token_down_proj],
            dim=-1
        )
        # print("hq_feature.shape",hq_feature.shape)

        return (self.hq_token2token(hq_feature)).reshape(-1,self.token_length).permute(1,0)

    
    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True) -> torch.Tensor:
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"

        if batch_first:
            x = x.permute(1, 0, 2)
        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=0)
            
        
        # B, C, H, W = handcrafted_feature.shape
        # handcrafted_feature = handcrafted_feature.view(B, C, H*W).permute(0, 2, 1)
        # joint_feature = torch.zeros_like(handcrafted_feature)
        # # if self.embedding_tune:
        # #     joint_feature += embedding_feature
        # if self.handcrafted_tune:
        #     joint_feature += handcrafted_feature
        # if hq_feature is not None:
        #     joint_feature += hq_feature
        # prompt = self.forward_delta_feat(embedding_feature, joint_feature, layer)

        tokens = self.get_tokens(x,layer)
        # print("tokens.shape",tokens.shape)
        delta_feat = self.forward_delta_feat(
            x,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        x = x + delta_feat
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=0)
        if batch_first:
            x = x.permute(1, 0, 2)
        return x
    

class Reins_Attention2(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=32,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.hq_token = hq_token
        self.use_softmax = use_softmax
        # self.hxw = patch_height * patch_width
        self.hq_token_down_proj = nn.Sequential(
            *[Adapter(in_features=hq_token.size(-1), mlp_ratio=0.125, add_last_layer=False)
              for _ in range(self.token_length)]
        )

        # assert patch_height is not None and patch_width is not None, "patch_height and patch_width should be provided"
        self.shared_up_proj = nn.Linear(
            in_features=int(hq_token.size(-1) * 0.125),
            out_features=int(embed_dims)
        )
        self.scale_init = scale_init
        self.create_model()


    def create_model(self):
        # self.learnable_tokens = nn.Parameter(
        #     torch.empty([self.num_layers, self.token_length, self.embed_dims])
        # )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        # self.hq_token2token = nn.Linear(len(self.hq_token_down_proj), self.token_length)
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_() 
    def get_mlps(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.mlp_token2feat, self.mlp_delta_f
        else:
            return self.mlp_token2feat, self.mlp_delta_f
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        mlp_token2feat, mlp_delta_f = self.get_mlps(layers)

        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn,
            mlp_token2feat(tokens),
        )
        delta_f = mlp_delta_f(delta_f + feats)
        return delta_f
    def get_tokens(self, x:torch.Tensor,layer: int) -> Tensor:

        hq_feature = torch.cat(
            [self.shared_up_proj(down_proj(self.hq_token)).unsqueeze(-1) for down_proj in self.hq_token_down_proj],
            dim=-1
        )
        # print("hq_feature.shape",hq_feature.shape)

        return (hq_feature).reshape(-1,self.token_length).permute(1,0)

    
    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True) -> torch.Tensor:
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"

        if batch_first:
            x = x.permute(1, 0, 2)
        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=0)
            
        
        # B, C, H, W = handcrafted_feature.shape
        # handcrafted_feature = handcrafted_feature.view(B, C, H*W).permute(0, 2, 1)
        # joint_feature = torch.zeros_like(handcrafted_feature)
        # # if self.embedding_tune:
        # #     joint_feature += embedding_feature
        # if self.handcrafted_tune:
        #     joint_feature += handcrafted_feature
        # if hq_feature is not None:
        #     joint_feature += hq_feature
        # prompt = self.forward_delta_feat(embedding_feature, joint_feature, layer)

        tokens = self.get_tokens(x,layer)
        # print("tokens.shape",tokens.shape)
        delta_feat = self.forward_delta_feat(
            x,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        x = x + delta_feat
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=0)
        if batch_first:
            x = x.permute(1, 0, 2)
        return x
def get_intervals(length,num):
    res = []
    for i in range(num):
        res.append(round(length / num ))
    return res
class Reins_Attention2_upd(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=32,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.hq_token = hq_token
        self.use_softmax = use_softmax
        self.hb = 32
        self.intervals = get_intervals(hq_token.size(-1)*0.125*self.hb,self.hb)
        # self.hxw = patch_height * patch_width
        self.hq_token_down_proj = nn.Sequential(
            *[Adapter(in_features=hq_token.size(-1), mlp_ratio=0.125*self.hb, add_last_layer=False)
              for _ in range(self.token_length // self.hb)]
        )

        # assert patch_height is not None and patch_width is not None, "patch_height and patch_width should be provided"
        self.shared_up_proj = nn.Linear(
            in_features=int(hq_token.size(-1) * 0.125),
            out_features=int(embed_dims)
        )
        self.scale_init = scale_init
        self.create_model()


    def create_model(self):
        # self.learnable_tokens = nn.Parameter(
        #     torch.empty([self.num_layers, self.token_length, self.embed_dims])
        # )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        # self.hq_token2token = nn.Linear(len(self.hq_token_down_proj), self.token_length)
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_() 
    def get_mlps(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.mlp_token2feat, self.mlp_delta_f
        else:
            return self.mlp_token2feat, self.mlp_delta_f
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        mlp_token2feat, mlp_delta_f = self.get_mlps(layers)

        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn,
            mlp_token2feat(tokens),
        )
        delta_f = mlp_delta_f(delta_f + feats)
        return delta_f
    def get_tokens(self, x:torch.Tensor,layer: int) -> Tensor:
        tmp_list = []
        for down_proj in self.hq_token_down_proj:
            # print(down_proj(self.hq_token).squeeze().shape)
            li2 = down_proj(self.hq_token).squeeze().split(self.intervals)
            for t in li2:
                tmp_list.append(self.shared_up_proj(t).unsqueeze(-1))
        hq_feature = torch.cat(
            tmp_list,
            dim=-1
        )
        

        return (hq_feature).reshape(-1,self.token_length).permute(1,0)

    
    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True) -> torch.Tensor:
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"

        if batch_first:
            x = x.permute(1, 0, 2)
        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=0)
            
        
        # B, C, H, W = handcrafted_feature.shape
        # handcrafted_feature = handcrafted_feature.view(B, C, H*W).permute(0, 2, 1)
        # joint_feature = torch.zeros_like(handcrafted_feature)
        # # if self.embedding_tune:
        # #     joint_feature += embedding_feature
        # if self.handcrafted_tune:
        #     joint_feature += handcrafted_feature
        # if hq_feature is not None:
        #     joint_feature += hq_feature
        # prompt = self.forward_delta_feat(embedding_feature, joint_feature, layer)

        tokens = self.get_tokens(x,layer)
        # print("tokens.shape",tokens.shape)
        delta_feat = self.forward_delta_feat(
            x,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        x = x + delta_feat
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=0)
        if batch_first:
            x = x.permute(1, 0, 2)
        return x
        
class Reins_Attention2_upd(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=32,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.hq_token = hq_token
        self.use_softmax = use_softmax
        self.hb = 32
        self.intervals = get_intervals(hq_token.size(-1)*0.125*self.hb,self.hb)
        # self.hxw = patch_height * patch_width
        self.hq_token_down_proj = nn.Sequential(
            *[Adapter(in_features=hq_token.size(-1), mlp_ratio=0.125*self.hb, add_last_layer=False)
              for _ in range(self.token_length // self.hb)]
        )
        

        # assert patch_height is not None and patch_width is not None, "patch_height and patch_width should be provided"
        self.shared_up_proj = nn.Linear(
            in_features=int(hq_token.size(-1) * 0.125),
            out_features=int(embed_dims)
        )
        self.scale_init = scale_init
        self.create_model()


    def create_model(self):
        # self.learnable_tokens = nn.Parameter(
        #     torch.empty([self.num_layers, self.token_length, self.embed_dims])
        # )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        # self.hq_token2token = nn.Linear(len(self.hq_token_down_proj), self.token_length)
        
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_() 
    def get_mlps(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.mlp_token2feat, self.mlp_delta_f
        else:
            return self.mlp_token2feat, self.mlp_delta_f
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        mlp_token2feat, mlp_delta_f = self.get_mlps(layers)

        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn,
            mlp_token2feat(tokens),
        )
        delta_f = mlp_delta_f(delta_f + feats)
        return delta_f
    def get_tokens(self, x:torch.Tensor,layer: int) -> Tensor:
        tmp_list = []
        for down_proj in self.hq_token_down_proj:
            li2 = down_proj(self.hq_token).squeeze().split(self.intervals)
            for t in li2:
                tmp_list.append(self.shared_up_proj(t).unsqueeze(-1))
        hq_feature = torch.cat(
            tmp_list,
            dim=-1
        )
        

        return (hq_feature).reshape(-1,self.token_length).permute(1,0)

    
    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True) -> torch.Tensor:
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"

        if batch_first:
            x = x.permute(1, 0, 2)
        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=0)
            
        
        # B, C, H, W = handcrafted_feature.shape
        # handcrafted_feature = handcrafted_feature.view(B, C, H*W).permute(0, 2, 1)
        # joint_feature = torch.zeros_like(handcrafted_feature)
        # # if self.embedding_tune:
        # #     joint_feature += embedding_feature
        # if self.handcrafted_tune:
        #     joint_feature += handcrafted_feature
        # if hq_feature is not None:
        #     joint_feature += hq_feature
        # prompt = self.forward_delta_feat(embedding_feature, joint_feature, layer)

        tokens = self.get_tokens(x,layer)
        delta_feat = self.forward_delta_feat(
            x,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        x = x + delta_feat
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=0)
        if batch_first:
            x = x.permute(1, 0, 2)
        return x
        
class Reins_Attention2_upd2(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=32,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.hq_token = hq_token
        self.use_softmax = use_softmax
        self.hb = 32
        self.mlp_ratio = 0.125
        self.intervals = get_intervals(hq_token.size(-1)*0.125*self.hb,self.hb)
        # self.hxw = patch_height * patch_width
        # self.hq_token_down_proj = nn.Sequential(
        #     *[Adapter(in_features=hq_token.size(-1), mlp_ratio=0.125*self.hb, add_last_layer=False)
        #       for _ in range(self.token_length // self.hb)]
        # )
        self.hq_token_down_proj_conv = nn.Conv2d(hq_token.size(-1),int(self.mlp_ratio*hq_token.size(-1)),1)
        self.activation_down_proj = nn.GELU()

        # assert patch_height is not None and patch_width is not None, "patch_height and patch_width should be provided"
        self.shared_up_proj = nn.Linear(
            in_features=int(hq_token.size(-1) * 0.125),
            out_features=int(embed_dims)
        )
        self.scale_init = scale_init
        self.create_model()


    def create_model(self):
        # self.learnable_tokens = nn.Parameter(
        #     torch.empty([self.num_layers, self.token_length, self.embed_dims])
        # )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        # self.hq_token2token = nn.Linear(len(self.hq_token_down_proj), self.token_length)
        
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_() 
    def get_mlps(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.mlp_token2feat, self.mlp_delta_f
        else:
            return self.mlp_token2feat, self.mlp_delta_f
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        mlp_token2feat, mlp_delta_f = self.get_mlps(layers)

        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn,
            mlp_token2feat(tokens),
        )
        delta_f = mlp_delta_f(delta_f + feats)
        return delta_f
    def get_tokens(self, x:torch.Tensor,layer: int) -> Tensor:
        # tmp_list = []
        # for down_proj in self.hq_token_down_proj:
        #     li2 = down_proj(self.hq_token).squeeze().split(self.intervals)
        #     for t in li2:
        #         tmp_list.append(self.shared_up_proj(t).unsqueeze(-1))
        # hq_feature = torch.cat(
        #     tmp_list,
        #     dim=-1
        # )
        tmp = self.hq_token.reshape(-1)
        tmp = tmp.repeat(self.token_length,1)
        tmp = tmp.unsqueeze(0)
        tmp = tmp.permute(2,0,1)
        

        # print("tmp.shape",tmp.shape)
        hq_feature = self.activation_down_proj(self.hq_token_down_proj_conv(tmp)).permute(1,2,0)
        # print("hq_feature.shape",hq_feature.shape)
        hq_feature = hq_feature.reshape(self.token_length,-1)
        # print("hq_feature.shape",hq_feature.shape)
        hq_feature = self.shared_up_proj(hq_feature)

        

        return hq_feature

    
    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True) -> torch.Tensor:
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"

        if batch_first:
            # H*W,B,C
            x = x.permute(1, 0, 2)
        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=0)
            
        
        # B, C, H, W = handcrafted_feature.shape
        # handcrafted_feature = handcrafted_feature.view(B, C, H*W).permute(0, 2, 1)
        # joint_feature = torch.zeros_like(handcrafted_feature)
        # # if self.embedding_tune:
        # #     joint_feature += embedding_feature
        # if self.handcrafted_tune:
        #     joint_feature += handcrafted_feature
        # if hq_feature is not None:
        #     joint_feature += hq_feature
        # prompt = self.forward_delta_feat(embedding_feature, joint_feature, layer)

        tokens = self.get_tokens(x,layer)
        delta_feat = self.forward_delta_feat(
            x,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale
        x = x + delta_feat
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=0)
        if batch_first:
            x = x.permute(1, 0, 2)
        return x
        
