import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import LlamaConfig
import clip
import numpy as np
from itertools import repeat
import collections.abc
from functools import partial
from colossalai.moe import SparseMLP
from colossalai.moe.manager import MOE_MANAGER


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)

dtype = torch.float32


#################################################################################
#                                 Sparse MOE Block                           #
#################################################################################


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
        # b n h w -> b patch**3 nhw//(patch**3) -> b nhw//patch**3 patch**3
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
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).type(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LanguageEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, model_clip, language_embedding=512):
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
        return text_feature.type(dtype)

    def forward(self, prompts):
        t_emb = self.get_text_feature(prompts)
        t_emb = self.mlp(t_emb)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


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
        self.mlp = SparseMLP(
            num_experts=8,
            hidden_size=128,
            intermediate_size=mlp_hidden_dim,
            router_top_k=2,
            router_capacity_factor_train=1.25,
            router_capacity_factor_eval=2.0,
            router_min_capacity=4,
            router_noisy_policy=None,
            router_drop_tks=True,
            mlp_activation="swiglu",
            mlp_gated=True,
            enable_load_balance=False,
            load_balance_tolerance=0.1,
            load_balance_beam_width=8,
            load_balance_group_swap_factor=0.4,
            enable_kernel=False,
            enable_comm_overlap=False,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
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
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size, bias=True)
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
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1,
                                                                                  4)  # batch num_heads token_num head_dim
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
            model_clip=None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_embedder = Patchify(patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.l_embedder = LanguageEmbedder(hidden_size, model_clip)
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

    def unpatchify(self, x, original_shape):
        """
        x: (N, T, patch_size**3) T = C H W //patch**3
        imgs: (N, C, H, W)
        """
        p = self.x_embedder.patch_size
        c, h, w = original_shape

        x = x.reshape(shape=(x.shape[0], c, h, w, p, p, p))
        x = torch.einsum('nchwpqe->ncphqwe', x)
        imgs = x.reshape(shape=(x.shape[0], c * p, h * p, h * p))
        return imgs

    def _calculate_router_loss(self, aux_loss: list = None, z_loss: list = None):
        if aux_loss is None or z_loss is None:
            aux_loss, z_loss = MOE_MANAGER.get_loss()
        assert len(aux_loss) == len(z_loss)
        aux_loss = 0.01 * sum(aux_loss) / len(aux_loss)
        z_loss = 0.0001 * sum(z_loss) / len(z_loss)
        return aux_loss, z_loss

    def forward(self, x, t, prompts1, prompts2, original_shape, target=None):
        """
        Forward pass of DiT.
        x: (N, 3C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        MOE_MANAGER.reset_loss()
        x = self.x_embedder(x)  # (N, T, D), where T = H * W * 2C / patch_size ** 3
        t = self.t_embedder(t)  # (N, D)
        c = t + self.l_embedder(prompts1) + self.l_embedder(prompts2)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 3)
        x = self.unpatchify(x, original_shape)  # (N, 2C, H, W)
        output = x[:, :x.shape[1] // 2, :, :]
        loss = None
        if target is not None:
            aux_loss, z_loss = self._calculate_router_loss()
            loss = aux_loss + z_loss + F.l1_loss(output, target, reduction="mean")
            print(loss.item(), aux_loss.item(), z_loss.item())
        return output, loss


class Router(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.router = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.router(x)

class MoEConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, ratio=0.5):
        super(MoEConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.router = Router(512, out_channels)
        self.top_k = int(ratio*out_channels)

    def forward(self, x, y):
        # Assuming router function returns a tensor of shape (batch_size, n_expert)
        self.scores = self.router(y)  # Shape: (batch_size, n_expert)

        # Get the top-K scores and their indices
        topk_scores, topk_indices = torch.topk(self.scores, self.top_k, dim=1)

        # Create a mask with 1s at the top-K indices and 0s elsewhere
        mask = torch.zeros_like(self.scores)
        mask.scatter_(1, topk_indices, 1)

        # Perform the convolution operation
        out = super(MoEConv, self).forward(x)  # Shape: (batch_size, out_channels * n_expert, D, H, W)

        # Apply the mask to the scores
        scores_expanded = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Shape: (batch_size, C, 1, 1, 1)
        out_selected = (out * scores_expanded).sum(dim=1)  # Shape: (batch_size, out_channels, D, H, W)

        return out_selected

class MOENetWork(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            hidden_size=1152,
            model_clip=None
    ):
        super().__init__()

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.l_embedder = LanguageEmbedder(hidden_size, model_clip)

    def forward(self, x, t, prompts1, prompts2):
        x = self.x_embedder(x)  # (N, T, D), where T = H * W * 2C / patch_size ** 3
        t = self.t_embedder(t)  # (N, D)
        c = t + self.l_embedder(prompts1) + self.l_embedder(prompts2)

       
        return output

#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_S_DEMO(model_clip, **kwargs):
    return DiT(model_clip=model_clip, depth=6, hidden_size=128, patch_size=4, num_heads=4, **kwargs)


import time

if __name__ == '__main__':
    device = "cuda:2"
    model_clip, _ = clip.load("ViT-B/32", device=device)  # only one 77 embeding is given, thus we can do this also?
    model = DiT_S_DEMO(model_clip).to(device)
    x = torch.randn((1, 8, 256, 256), device=device,
                    dtype=torch.float32)  # 输入xt,ms,duplicate pan-ms 三部分  输入token比较长，重建只在一部分token上计算loss
    y = "this is a test"
    y = clip.tokenize(y).to(device)
    time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=1)).to(device)
    original_tuple = (8, 256, 256)
    print(time_in, time_in.shape)
    with torch.no_grad():
        output, _ = model(x, time_in, y, y)
    t2 = time.time()
    print(output.shape)
