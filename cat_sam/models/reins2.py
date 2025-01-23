
from torch import nn, Tensor
import torch
import math
class Reins_Attention4(nn.Module):
    def __init__(self, embed_dims: int,  num_layers: int, patch_size:int ,token_length:int=100,embed_dims_ratio:int=32,use_softmax: bool = True,hq_token: torch.Tensor = None, 
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
        self.hb = 32
        self.mlp_ratio = 0.125
        self.down_proj = nn.Linear(
            in_features=hq_token.size(-1),
            out_features=int(embed_dims*self.embed_dims_ratio)
        )
        self.gelu = nn.GELU()
        self.up_proj = nn.Linear(
            in_features=int(embed_dims*self.embed_dims_ratio),
            out_features=int(embed_dims)
        )
        self.freq_nums = 0.25
        self.scale_init = scale_init
        self.create_model()


    def create_model(self):
        # self.learnable_tokens = nn.Parameter(
        #     torch.empty([self.num_layers, self.token_length, self.embed_dims])
        # )
        self.A = nn.Parameter(torch.empty([1, self.token_length, round(self.embed_dims*self.embed_dims_ratio)]))
        self.B = nn.Parameter(torch.empty([self.num_layers,round(self.embed_dims*self.embed_dims_ratio),self.token_dim]))

        
        # self.hq_token2token = nn.Linear(len(self.hq_token_down_proj), self.token_length)
        
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        
        
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
        # print("self.A.shape",self.A[0].shape)
        # print("self.B.shape",self.B[layer].shape)
        B = self.B[layer]
        B = self.f(B)
        tokens = self.A[0] @ B

        return tokens

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums)
        return self.prompt_generator(x)
    
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
    
    def forward(self,x: torch.Tensor,layer:int,batch_first=False, has_cls_token=True) -> torch.Tensor:
        assert layer >= 0 or layer < self.num_layers , "layer should be in range of 0 to num_layers"

        if batch_first:
            # H*W,B,C
            x = x.permute(1, 0, 2)
        if has_cls_token:
            cls_token, x = torch.tensor_split(x, [1], dim=0)
            
        handcrafted_feature = self.init_handcrafted(x)

        B, C, H, W = handcrafted_feature.shape
        handcrafted_feature = handcrafted_feature.view(B, C, H*W).permute(0, 2, 1)
        joint_feature = torch.zeros_like(handcrafted_feature)
        # if self.embedding_tune:
        #     joint_feature += embedding_feature
        if self.handcrafted_tune:
            joint_feature += handcrafted_feature
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
        