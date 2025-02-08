
from sympy import N
from torch import nn, Tensor
import torch
import math
import torch.nn.functional as F
from cat_sam.models.module_lib import PatchEmbed2
import collections.abc as container_abcs
from itertools import repeat

def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

class PatchEmbed2(nn.Module):
    """
    Adapted from
    https://github.com/tianrun-chen/SAM-Adapter-PyTorch/blob/60bd2770b1c4fcb38d98822cae9a17781601ff9e/models/mmseg/models/sam/image_encoder.py#L340
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        return x


class EVP(nn.Module):
    def __init__(self,img_size=224, patch_size=16, in_chans=3, embed_dim=768,freq_nums=0.25):
        super().__init__()
        self.patch_embed = PatchEmbed2(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.freq_nums = freq_nums
        
    def forward(self,x):
        x = self.fft(x,self.freq_nums)
        return self.patch_embed(x)

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums)
        return self.patch_embed(x)
    
    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        fft = fft * (1 - mask)
        # fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv

class EVP2(nn.Module):
    def __init__(self,img_size=224, patch_size=16, in_chans=3, embed_dim=768,freq_nums=0.25):
        super().__init__()
        self.patch_embed = PatchEmbed2(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.freq_nums = freq_nums
        
    def forward(self,x):
        x = self.fft(x,self.freq_nums)
        return self.patch_embed(x)

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums)
        return self.patch_embed(x)
    
    def fft(self, x, rate):
        # the smaller rate, the smoother; the larger rate, the darker
        # rate = 4, 8, 16, 32
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1

        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        # mask[fft.float() > self.freq_nums] = 1
        # high pass: 1-mask, low pass: mask
        fft = fft * (1 - mask)
        # fft = fft * mask
        fr = fft.real
        fi = fft.imag

        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real

        inv = torch.abs(inv)

        return inv


class Reins_Attention4(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=1,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.hq_token = hq_token
        self.token_dim = hq_token.size(-1)
        self.use_softmax = use_softmax
        self.embed_dims_ratio = embed_dims_ratio
        self.mlp_ratio = 0.125        
        self.freq_nums = 0.25
        self.scale_init = scale_init
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
        self.A = nn.Parameter(torch.empty([1, self.token_length, round(self.embed_dims*self.embed_dims_ratio)]))
        self.B = nn.Parameter(torch.empty([self.num_layers,round(self.embed_dims*self.embed_dims_ratio),self.token_dim]))
           
        
        self.token2hq = nn.Linear(3*self.token_length*self.embed_dims, self.token_dim)
        
        self.apply(self._init_weights)
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
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
    def get_hq_token(self) -> Tensor:
        # L,M,C-> M,C,L
        tokens = self.get_tokens(-1).permute(1, 2, 0)
        # print(tokens.shape)
        
        max_token = F.max_pool1d(tokens, kernel_size=self.num_layers)
        avg_token = F.avg_pool1d(tokens, kernel_size=self.num_layers)
        last_token = tokens[:, :, -1].unsqueeze(-1)
        # print(max_token.shape,avg_token.shape,last_token.shape)
        concat = torch.cat([max_token, avg_token, last_token], dim=-1)
        concat = concat.flatten()
        hq_token = self.token2hq(concat).reshape(1,-1)
        # print("hq_token")
        # print(hq_token)
        return hq_token

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
    def f(self,B):
        return self.up_proj(self.gelu(self.down_proj(B)))
        
    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            B = self.B
            B = self.f(B)
            tokens = self.A @ B
            # print("tokens -1",tokens.shape)
            return tokens

        else:
            B = self.B[layer]
            B = self.f(B)
            tokens = self.A[0] @ B

            return tokens

    
    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True,evp_feature=None) -> torch.Tensor:
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"
        if evp_feature is not None:
            B, C, H, W = evp_feature.shape
            evp_feature = evp_feature.view(B, C, -1).permute(0,2,1)
            x = x + evp_feature
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
        )
        delta_feat = delta_feat * self.scale
        x = x + delta_feat
        if has_cls_token:
            x = torch.cat([cls_token, x], dim=0)
        if batch_first:
            x = x.permute(1, 0, 2)
        return x
        
class Reins_Attention5(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=1,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.hq_token = hq_token
        self.token_dim = hq_token.size(-1)
        self.use_softmax = use_softmax
        self.embed_dims_ratio = embed_dims_ratio
        self.mlp_ratio = 0.125

        
        self.freq_nums = 0.25
        self.scale_init = scale_init
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
        self.A = nn.Parameter(torch.empty([1, self.token_length, round(self.embed_dims*self.embed_dims_ratio)]))
        self.B = nn.Parameter(torch.empty([self.num_layers,round(self.embed_dims*self.embed_dims_ratio),self.token_dim]))
        
        
        self.apply(self._init_weights)
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
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
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int,evp_feature=None) -> Tensor:
        if evp_feature is not None:
            # B = evp_feature.shape[0]
            # # H*W,B,C -> B,H*W,C
            # feats = feats.permute(1,0,2)
            # attn = []
            # for b in range(B):
            #     print(feats[b].shape)
            #     print((evp_feature[b]+tokens).shape)
                
            #     attn.append((feats[b] @ (evp_feature[b]+tokens).permute(1,0)).unsqueeze(0))
            #     # attn.append(torch.einsum("nc,mc->nm", feats[b], evp_feature[b]+tokens).unsqueeze(0))
            # B,H*W,m -> H*W,B,m
            b = 0
            attn = (feats[b] @ (evp_feature[b]+tokens).permute(1,0)).unsqueeze(0).permute(1,0,2)
            

        else:
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
    def f(self,B):
        return self.up_proj(self.gelu(self.down_proj(B)))
        
    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            B = self.B
            B = self.f(B)
            tokens = self.A @ B
            return tokens

        else:
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
        
        
class Reins_Attention6(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=1,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False,connect_with_hq_token=True,c_hq_num:int=1) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.token_length = token_length
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.hq_token = hq_token
        self.token_dim = hq_token.size(-1)
        self.use_softmax = use_softmax
        self.embed_dims_ratio = embed_dims_ratio
        self.mlp_ratio = 0.125
        self.connect_with_hq_token = connect_with_hq_token

        
        self.freq_nums = 0.25
        self.scale_init = scale_init
        self.c_hq_num = c_hq_num
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
        
        self.A = nn.Parameter(torch.empty([1, self.token_length, round(self.embed_dims*self.embed_dims_ratio)]))
        if self.connect_with_hq_token:
            self.B = nn.Parameter(torch.empty([self.num_layers + 1,round(self.embed_dims*self.embed_dims_ratio)-self.c_hq_num,self.token_dim]))
        else:
            self.B = nn.Parameter(torch.empty([self.num_layers + 1,round(self.embed_dims*self.embed_dims_ratio),self.token_dim]))
        
        self.apply(self._init_weights)
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
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


    def get_mlps(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.mlp_token2feat, self.mlp_delta_f
        else:
            return self.mlp_token2feat, self.mlp_delta_f
    
    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int,evp_feature=None) -> Tensor:
        if evp_feature is not None:
            b = 0
            attn = (feats[b] @ (evp_feature[b]+tokens).permute(1,0)).unsqueeze(0).permute(1,0,2)
        else:
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
    def f(self,B):
        return self.up_proj(self.gelu(self.down_proj(B)))

        
    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            B = self.B
            B = self.f(B)
            tokens = self.A @ B
            return tokens
        else:
            B = self.B[layer]
            if self.connect_with_hq_token:
                B = torch.concat([self.hq_token]*self.c_hq_num+[B],dim=0)
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
    

class Reins_Attention7(Reins_Attention6):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=1,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False,connect_hq_token=True,c_hq_num:int=1) -> None:
        super().__init__(
            embed_dims,num_layers,patch_size,token_length,embed_dims_ratio,use_softmax,hq_token,scale_init,zero_mlp_delta_f,connect_with_hq_token=connect_hq_token
            ,c_hq_num=c_hq_num
        )

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            raise TypeError
            return None
        else:
            B = self.B[layer]
            if self.connect_with_hq_token:
                B = torch.concat([self.hq_token]*self.c_hq_num+[B],dim=0)
            # B = self.f(B)
            tokens = self.A[0] @ B
            return self.f(tokens)

class Reins_Attention8(Reins_Attention6):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=1,use_softmax: bool = True,hq_token: torch.Tensor = None, 
                 scale_init: float = 0.001, zero_mlp_delta_f: bool = False,connect_hq_token=True,c_hq_num:int=1) -> None:
        
        super().__init__(
            embed_dims,num_layers,patch_size,token_length,embed_dims_ratio,use_softmax,hq_token,scale_init,zero_mlp_delta_f,connect_with_hq_token=connect_hq_token
            ,c_hq_num=c_hq_num
        )
        if self.connect_with_hq_token:
            self.linear_hq_token = nn.Linear(self.token_dim,self.embed_dims)



    def create_model(self):
        super().create_model()
        del self.B
        if self.connect_with_hq_token:
            self.B = nn.Parameter(torch.empty([self.num_layers + 1,round(self.embed_dims*self.embed_dims_ratio)-self.c_hq_num,self.embed_dims]))
        else:
            self.B = nn.Parameter(torch.empty([self.num_layers + 1,round(self.embed_dims*self.embed_dims_ratio),self.embed_dims]))

        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            raise TypeError
            return None
        else:

            
            B = self.B[layer]
            if self.connect_with_hq_token:
                B = torch.concat([self.linear_hq_token(self.hq_token)]*self.c_hq_num+[B],dim=0)
            tokens = self.A[0] @ B
            return tokens