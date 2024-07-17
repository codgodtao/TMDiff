#################################################################################
#                                 MOE_Conv                                 #
#################################################################################
import torch
import math
from torch import autograd, nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


def get_device(x):
    gpu_idx = x.get_device()
    return f"cuda:{gpu_idx}" if gpu_idx >= 0 else "cpu"


class Router(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.router = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.router(x)


class MoEConv(nn.Conv3d):
    def __init__(self, in_channels, out_channels, hidden_size, kernel_size=3, stride=1, padding=1):
        super(MoEConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.router = Router(hidden_size, out_channels)
        self.softmax = nn.LeakyReLU()

    def forward(self, x, y):
        # split prompt and timesteps
        self.scores = self.softmax(self.router(y))  # Shape: (batch_size, n_expert)
        # Perform the convolution operation
        out = super(MoEConv, self).forward(x)  # Shape: (batch_size, out_channels * n_expert, D, H, W)
        # Apply scores to select channels
        #细粒度专家分割与共享专家，expert得到的channel进行分析
        scores_expanded = self.scores.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Shape: (batch_size, C, 1, 1, 1)
        out_selected = out * scores_expanded  # Shape: (batch_size, out_channels, D, H, W)

        return out_selected


##########################################################################
## Layer Norm


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c n h w -> b (n h w) c')


def to_4d(x, n, h, w):
    return rearrange(x, 'b (n h w) c -> b c n h w', n=n, h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        n, h, w = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), n, h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Attention3D(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention3D, self).__init__()
        self.num_heads = num_heads

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, n, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) n h w -> b head c (n h w)', head=self.num_heads)  # n*h*w就是 hidden dim
        k = rearrange(k, 'b (head c) n h w -> b head c (n h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) n h w -> b head c (n h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        d_head = n * h * w
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
        attn = attn.softmax(dim=-1)  # b head c*c的attention score

        out = (attn @ v)

        out = rearrange(out, 'b head c (n h w) -> b (head c) n h w', head=self.num_heads, n=n, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Transformer3DBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type, hidden_size):
        super(Transformer3DBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention3D(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MoEConv(dim, dim, hidden_size)

    def forward(self, x, y):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x), y)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # b c n h w->downsample b c//2 n h w-> bn c//2 h w -> bn c*2 h/2 w/2  -> b c**2 n h/2 w/2
        self.body = nn.Conv3d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.unshuffle = nn.PixelUnshuffle(2)

    def forward(self, x):
        b, c, n, h, w = x.size()
        # print(x.size())
        x = self.body(x)
        # print(x.size(),x.transpose(1,2).shape)
        x = x.transpose(1, 2).reshape(b * n, c // 2, h, w)
        x = self.unshuffle(x).reshape(b, n, c * 2, h // 2, w // 2).transpose(2, 1)
        return x


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Conv3d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        # #b c n h w->upsample b c**2 n h w-> bn c**2 h w -> bn c//2 h*2 w*2  -> b c**2 n h/2 w/2
        b, c, n, h, w = x.size()
        x = self.body(x)
        x = x.transpose(2, 1).reshape(b * n, c * 2, h, w)
        x = self.shuffle(x).reshape(b, n, c // 2, h * 2, w * 2).transpose(2, 1)
        return x


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


dtype = torch.float32


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


##########################################################################
##---------- Restormer -----------------------
class MOENetWork(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 hidden_size=128,
                 dim=48,
                 heads=None,
                 bias=False,
                 model_clip=None,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=True  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(MOENetWork, self).__init__()

        if heads is None:
            heads = [1, 2, 4, 8]
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.l_embedder = LanguageEmbedder(hidden_size, model_clip)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = Transformer3DBlock(dim=dim, num_heads=heads[0], bias=bias, LayerNorm_type=LayerNorm_type,hidden_size=hidden_size)

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = Transformer3DBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], bias=bias,
                                                 LayerNorm_type=LayerNorm_type,hidden_size=hidden_size)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3  *2
        self.encoder_level3 = Transformer3DBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                 bias=bias, LayerNorm_type=LayerNorm_type,hidden_size=hidden_size)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4  *4
        self.latent = Transformer3DBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                                         bias=bias, LayerNorm_type=LayerNorm_type,hidden_size=hidden_size)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3      *8
        self.reduce_chan_level3 = nn.Conv3d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = Transformer3DBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                                                 bias=bias, LayerNorm_type=LayerNorm_type,hidden_size=hidden_size)

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = Transformer3DBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                                                 bias=bias, LayerNorm_type=LayerNorm_type,hidden_size=hidden_size)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = Transformer3DBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                                 bias=bias, LayerNorm_type=LayerNorm_type,hidden_size=hidden_size)

        self.refinement = Transformer3DBlock(dim=int(dim * 2 ** 1), num_heads=heads[0],
                                             bias=bias, LayerNorm_type=LayerNorm_type,hidden_size=hidden_size)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv3d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, t, prompts1, prompts2):
        # b 3 c h w input --> b 1 c h w --> b c h w for loss function
        t = self.t_embedder(t)  # (N, D)
        y = t + self.l_embedder(prompts1) + self.l_embedder(prompts2)  # 后续router计算分数
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, y)
        # print(inp_enc_level1.shape, out_enc_level1.shape)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        # print(inp_enc_level2.shape)
        out_enc_level2 = self.encoder_level2(inp_enc_level2, y)
        # print(inp_enc_level2.shape, out_enc_level2.shape)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3, y)
        # print(inp_enc_level3.shape, out_enc_level3.shape)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4, y)
        # print(inp_enc_level4.shape, latent.shape)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, y)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, y)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, y)

        out_dec_level1 = self.refinement(out_dec_level1, y)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img
        #b 1 c h w
        return out_dec_level1


# # Example usage
# import clip
# import numpy as np
# device = "cuda:0"
# model_clip, _ = clip.load("ViT-B/32", device=device)  # only one 77 embeding is given, thus we can do this also?
# y = "this is a test"
# y = clip.tokenize(y).to(device)
# time_in = torch.from_numpy(np.random.randint(1, 1000 + 1, size=1)).to(device)
# model = MOENetWork(inp_channels=3, out_channels=1, dim=48,model_clip=model_clip).to(device)
# img = torch.randn(10, 3, 8, 64, 64).to(device)
# output = model(img, time_in,y,y)
# print(output)
