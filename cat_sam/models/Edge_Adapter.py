
import token
from sympy import N
from torch import nn, Tensor
import torch
import math
import torch.nn.functional as F


class Base_Edge_Adapter(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=1,use_softmax: bool = True,token_dim:int=256,
                 scale_init: float = 0.001) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        
        self.use_softmax = use_softmax
        self.embed_dims_ratio = embed_dims_ratio
        self.mlp_ratio = 0.125
        self.token_dim = token_dim

        self.scale_init = scale_init
        self.r = round(self.token_dim*self.embed_dims_ratio)
        self.create_model()
        

    def create_model(self):
        self.down_proj = nn.Linear(
            in_features=self.token_dim,
            out_features=int(self.embed_dims*self.mlp_ratio)
        )
        self.gelu = nn.GELU()
        self.up_proj = nn.Linear(
            in_features=int(self.embed_dims*self.mlp_ratio),
            out_features=int(self.embed_dims)
        )
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        self.A = nn.Parameter(torch.empty([1, self.token_length, self.r]))

        self.B = nn.Parameter(torch.empty([self.num_layers + 1,self.r,self.token_dim]))
        
        self.apply(self._init_weights)
        
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        
        nn.init.kaiming_normal_(self.A, a=math.sqrt(5))
        
        nn.init.kaiming_normal_(self.B, a=math.sqrt(5))
        
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
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
        elif isinstance(m, nn.Parameter):
            nn.init.kaiming_normal_(m, a=math.sqrt(5))


    def get_mlps(self, layer: int) -> Tensor:
        
        return self.mlp_token2feat, self.mlp_delta_f
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int,evp_feature=None) -> Tensor:
        if evp_feature is not None:
            token_with_evp = (evp_feature + tokens).permute(0,2,1)
            attn = torch.bmm(feats.permute(1,0,2), token_with_evp).permute(1,0,2)
        else:
            attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        mlp_token2feat, mlp_delta_f = self.get_mlps(layers)
        if self.use_softmax:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        self.attn = attn.clone()
        
        if evp_feature is not None:
            delta_f = torch.bmm(attn.permute(1,0,2), mlp_token2feat(token_with_evp.permute(0,2,1))).permute(1,0,2)
        else:
            delta_f = torch.einsum(
                "nbm,mc->nbc",
                attn,
                mlp_token2feat(tokens),
            )
        delta_f = mlp_delta_f(delta_f + feats)
        return delta_f
    
    def f(self,B):
        return self.up_proj(self.gelu(self.down_proj(B)))

    def get_attention(self):
        return self.attn
    def get_tokens(self, layer: int) -> Tensor:
        B = self.B[layer]
        B = self.f(B)
        tokens = self.A[0] @ B

        return tokens

    
    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True,evp_feature=None) -> torch.Tensor:
        
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"
        if batch_first:
            # H*W,B,C
            x = x.permute(1, 0, 2)
        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=0)

        tokens = self.get_tokens(layer)

        delta_feat = self.forward_delta_feat(
            x,
            tokens,
            layer,
            evp_feature
        )
        delta_feat = delta_feat * self.scale
        x = x + delta_feat
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=0)
        if batch_first:
            x = x.permute(1, 0, 2)
        return x
    

class Edge_Adapter(Base_Edge_Adapter):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=1,use_softmax: bool = True,token_dim:int=256,
                 scale_init: float = 0.001) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_layers=num_layers,
            patch_size=patch_size,
            token_length=token_length,
            embed_dims_ratio=embed_dims_ratio,
            use_softmax=use_softmax,
            token_dim=token_dim,
            scale_init=scale_init
        )
    def get_tokens(self, layer: int) -> Tensor:
        B = self.B[layer]
        tokens = self.f(self.A[0] @ B)
        return tokens

class Edge_Adapter2(Base_Edge_Adapter):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=1,use_softmax: bool = True,token_dim:int=256,
                 scale_init: float = 0.001) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_layers=num_layers,
            patch_size=patch_size,
            token_length=token_length,
            embed_dims_ratio=embed_dims_ratio,
            use_softmax=use_softmax,
            token_dim=token_dim,
            scale_init=scale_init
        )



    def create_model(self):
        super().create_model()
        del self.B

        self.B = nn.Parameter(torch.empty([self.num_layers + 1,round(self.embed_dims*self.embed_dims_ratio),self.embed_dims]))

        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_tokens(self, layer: int) -> Tensor:
        B = self.B[layer]
        tokens = self.A[0] @ B
        return tokens

class Local_Enforcement(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int,embed_dims_ratio=0.125) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.embed_dims_ratio = embed_dims_ratio
        self.num_layers = num_layers

        
        self.create_model()

    def create_model(self):
        
        for i in range(self.num_layers):
            setattr(self, f"down_proj_{i}", nn.Linear(
                in_features=self.embed_dims,
                out_features=int(self.embed_dims*self.embed_dims_ratio)
            ))
            
        self.gelu = nn.GELU()
        self.up_proj = nn.Linear(
            in_features=int(self.embed_dims*self.embed_dims_ratio),
            out_features=int(self.embed_dims)
        )
        self.scale = nn.Parameter(torch.tensor(0.1),requires_grad=True)
        self.token = nn.Parameter(torch.zeros((1,int(self.embed_dims*self.embed_dims_ratio))),requires_grad=True)
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
        elif isinstance(m, nn.Parameter):
            nn.init.kaiming_uniform_(m, a=math.sqrt(5))


    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True) -> torch.Tensor:
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"
        if batch_first:
            # H*W,B,C
            x = x.permute(1, 0, 2)
        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=0)
        
        token = self.token
        down_proj = getattr(self, f"down_proj_{layer}")
        delta_feat = self.up_proj(self.gelu((down_proj(x) + token)))
        x = delta_feat * self.scale + x
        
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=0)
        if batch_first:
            x = x.permute(1, 0, 2)
        return x