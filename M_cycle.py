# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://kkgithub.com/openai/glide-text2im
# MAE: https://kkgithub.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                 Sparse MOE Block                           #
#################################################################################
'''
openMOE
# Install ColossalAI
git clone --branch my_openmoe https://github.com/Orion-Zheng/ColossalAI.git
pip install ./ColossalAI
python -m pip install -r ./ColossalAI/examples/language/openmoe/requirements.txt

'''

from colossalai.moe.layers import SparseMLP
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import get_activation, set_moe_args

"""
    MoE related arguments.
    It inserts the MoE arguments into the Llama config.
    Args:
        config (LlamaConfig): Transformers Llama config.
        num_experts (int, optional): Number of experts.
        moe_layer_interval (int, optional): The interval moe layer.
        router_topk (int, optional): Moe router top k. Defaults to 2.
        router_capacity_factor_train (float, optional): Moe router max capacity for train. Defaults to 1.25.
        router_capacity_factor_eval (float, optional): Moe router max capacity for eval. Defaults to 2.0.
        router_min_capacity (int, optional): Moe router min capacity. Defaults to 4.
        router_noisy_policy (str, optional): Moe router noisy policy. You can choose [Jitter, Gaussian, None]. Defaults to None.
        router_drop_tks (bool, optional): Whether moe router drop tokens which exceed max capacity. Defaults to True.
        router_aux_loss_factor (float, optional): Moe router aux loss. You can refer to STMoE for details. Defaults to 0.01.
        router_z_loss_factor (float, optional): Moe router z loss. You can refer to STMoE for details. Defaults to 0.01.
        mlp_gated (bool, optional): Use gate in mlp. Defaults to True.
        label_smoothing (float, optional): Label smoothing. Defaults to 0.001.
        z_loss_factor (float, optional): The final outputs' classification z loss factor. Defaults to 0.01.
        enable_load_balance (bool, optional): Expert load balance. Defaults to False.
        load_balance_tolerance (float, optional): Expert load balance search's difference tolerance. Defaults to 0.1.
        load_balance_beam_width (int, optional): Expert load balance search's beam width. Defaults to 8.
        load_balance_group_swap_factor (float, optional): Expert load balance group swap factor. Longer value encourages less swap. Defaults to 0.4.
        enable_kernel (bool, optional): Use kernel optimization. Defaults to False.
        enable_comm_overlap (bool, optional): Use communication overlap for MoE. Recommended to enable for muiti-node training. Defaults to False.
        enable_hierarchical_alltoall (bool, optional): Use hierarchical alltoall for MoE. Defaults to False.
"""
    
class LitoQuMoeDecoderModel(nn.Module):
    def __init__(self, config: LlamaConfig, moe: bool):
        super().__init__()
        self.mlp = SparseMLP(
                num_experts=config.num_experts,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                router_top_k=config.router_topk,
                router_capacity_factor_train=config.router_capacity_factor_train,
                router_capacity_factor_eval=config.router_capacity_factor_eval,
                router_min_capacity=config.router_min_capacity,
                router_noisy_policy=config.router_noisy_policy,
                router_drop_tks=config.router_drop_tks,
                mlp_activation=config.hidden_act,
                mlp_gated=config.mlp_gated,
                enable_load_balance=config.enable_load_balance,
                load_balance_tolerance=config.load_balance_tolerance,
                load_balance_beam_width=config.load_balance_beam_width,
                load_balance_group_swap_factor=config.load_balance_group_swap_factor,
                enable_kernel=config.enable_kernel,
                enable_comm_overlap=config.enable_comm_overlap,
            )
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        target = None
    ):
        MOE_MANAGER.reset_loss()
        
        loss = None
        if target!=None:
            aux_loss, z_loss = self._calculate_router_loss()
            loss = aux_loss + z_loss +F.mse_loss(hidden_states,target)

            return hidden_states,loss
        return hidden_states,loss
    
    def _calculate_router_loss(self, aux_loss: list = None, z_loss: list = None):
        if aux_loss is None or z_loss is None:
            aux_loss, z_loss = MOE_MANAGER.get_loss()
        assert len(aux_loss) == len(z_loss)
        aux_loss = self.config.router_aux_loss_factor * sum(aux_loss) / len(aux_loss)
        z_loss = self.config.router_z_loss_factor * sum(z_loss) / len(z_loss)
        return aux_loss, z_loss

import math

def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings

def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)


    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了



    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos

    return q, k

#q, k = RoPE(q, k)
#yarn: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/modeling_llama_yarn.py#L228
#################################################################################
#                                 condition modulate                           #
#################################################################################


class Patchify(nn.Module):
    def __init__(self, patch_size=2, embedding_dim=32):
        super(Patchify, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        # 卷积层用于提取patch
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=embedding_dim, kernel_size=patch_size,
                               stride=patch_size)

    def forward(self, x):
        #b n h w -> b patch**3 nhw//(patch**3) -> b nhw//patch**3 patch**3
        x = x.unsqueeze(1)
        b, c, n, h, w = x.shape

        # 确保高度和宽度可以被patch_size整除
        assert h % self.patch_size == 0 and w % self.patch_size == 0 and n % self.patch_size == 0, "Height and Width must be divisible by patch size"

        # 使用卷积操作来提取patch并进行embedding
        patches = self.conv1(x)  # 形状为 (b, embedding_dim, n/patch_size, h/patch_size, w/patch_size)
        # 调整形状为 (b, nhw//(patch**3), embedding_dim)
        patches = patches.flatten(2).transpose(1, 2)

        return patches

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://kkgithub.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LanguageEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, model_clip,language_embedding=512):
        super().__init__()
        self.model_clip = model_clip.eval()
        self.mlp = nn.Sequential(
            nn.Linear(language_embedding, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature


    def forward(self, prompts):
        #clip model
        t_emb = self.get_text_feature(prompts)
        t_emb = self.mlp(t_emb)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        #给定condition，输出6个调制向量，3个一组控制attention和MLP
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size* patch_size * patch_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Attention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        patch_size=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        model_clip = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
    

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_embedder = Patchify(patch_size,hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.l_embedder = LanguageEmbedder(hidden_size,model_clip)
        # self.layers = nn.ModuleList(
        #     [
        #         OpenMoeDecoderLayer(config, moe=True if (i + 1) % config.moe_layer_interval == 0 else False)
        #         for i in range(config.num_hidden_layers)
        #     ]
        # )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.conv1.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.conv1.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.l_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.l_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x,original_shape):
        """
        #4*4*4=64
        x: (N, T, patch_size**2*outchannel)
        imgs: (N, C, H, W)
        """
        print(x.shape)
        p = self.x_embedder.patch_size
        c,h,w = original_shape

        x = x.reshape(shape=(x.shape[0],c, h, w, p, p, p))
        x = torch.einsum('nchwpqe->ncphqwe', x)
        imgs = x.reshape(shape=(x.shape[0], c*p, h * p, h * p))
        return imgs
    
    def forward(self, x, t, prompts,original_shape):
        """
        Forward pass of DiT.
        x: (N, 3C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)                    # (N, T, D), where T = H * W * 3C / patch_size ** 3
        t = self.t_embedder(t)                   # (N, D)
        l = self.l_embedder(prompts)
        c = t+l
        print(x.shape,t.shape,l.shape)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 3)
        x = self.unpatchify(x,original_shape)                   # (N, 3C, H, W)
        return x[:, :x.shape[1] // 3, :, :]


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_S_DEMO(model_clip,**kwargs):
    return DiT(model_clip=model_clip, depth=2, hidden_size=14, patch_size=4, num_heads=2, **kwargs)


if __name__ == '__main__':

    # channels=None, embed_dim=256, inter_dim=32, spectrun_num=4
    import clip
    device = 'cpu'
    model_clip, _ = clip.load("ViT-B/32", device=device) #only one 77 embeding is given, thus we can do this also?
    model_path = "OrionZheng/openmoe-base"
    config = AutoConfig.from_pretrained(model_path)

    model = DiT_S_DEMO(model_clip)
    optim_params = []
    for name, param in model.named_parameters():
        # print(name, param.shape)
        optim_params.append(param)
    print(sum(p.numel() for p in optim_params) / (1024.0 * 1024.0))


    x = torch.randn((2, 24, 512, 512)) #输入xt,ms,duplicate pan-ms 三部分  输入token比较长，重建只在一部分token上计算loss
    y = ["this is a test","this is a test"]
    y = clip.tokenize(y)
    import time
    import numpy as np

    time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=2))
    t1 = time.time()
    original_tuple  = (24,512,512)
    shape = (x // 4 for x in original_tuple)
    output = model(x, time_in, y,shape)
    t2 = time.time()
    # print(t2 - t1)
    print(output.shape)



    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # 生成随机目标数据
    # targets = torch.randn(batch_size,input_size, hidden_size)

    # # 训练循环
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     model.train()
    #     optimizer.zero_grad()
        
    #     outputs,loss = model(inputs,targets)
        
    #     loss.backward()
    #     optimizer.step()
        
    #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    # # 推理
    # model.eval()
    # with torch.no_grad():
    #     test_inputs = torch.randn(batch_size, input_size,hidden_size)
    #     test_outputs,_ = model(test_inputs)
    #     print("Inference output:", test_outputs.shape)
