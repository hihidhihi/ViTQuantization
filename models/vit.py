import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from time import time
import torch.nn.functional as F


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def ste_round(x):
    return torch.round(x) - x.detach() + x


def quantize(q, quat):
    sq = (q.max() - q.min()) / (2 ** quat - 1)
    zq = torch.round(2 ** quat - 1 - q.max() / sq)
    q = torch.round(q / sq + zq)
    return q, sq, zq


def fake_quantize(q, quat):
    sq = ((q.max() - q.min()) / (2 ** quat - 1)).detach()
    zq = torch.round(2 ** quat - 1 - q.max() / sq).detach()
    return sq * (ste_round(q / sq + zq) - zq)


def trainsz_quantize(q, sq, zq, quat):
    z = ste_round(zq)
    return sq * (torch.clamp(ste_round(q / sq) + z, 0, 2 ** quat - 1) - z)


def sz_init(q, quat):
    sq = ((q.max() - q.min()) / (2 ** quat - 1)).detach()
    zq = torch.round(2 ** quat - 1 - q.max() / sq).detach()
    return nn.Parameter(sq), nn.Parameter(zq)


# classes
def sz_quantize(q, sq, zq, quat):
    return torch.clamp(torch.round(q / sq) + torch.round(zq), 0, 2 ** quat - 1)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., quat=4):
        super().__init__()
        self.net = nn.Sequential(
            #nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            #nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.flag = 0
        self.quat = quat

        self.ww1 = nn.Parameter(torch.empty((hidden_dim, dim)))
        self.bb1 = nn.Parameter(torch.empty(hidden_dim))
        self.ww2 = nn.Parameter(torch.empty((dim, hidden_dim)))
        self.bb2 = nn.Parameter(torch.empty(dim))

        self.sw1, self.zw1 = sz_init(self.ww1, self.quat)
        self.sx1 = nn.Parameter(torch.tensor(0.))
        self.zx1 = nn.Parameter(torch.tensor(0.))
        self.sw2, self.zw2 = sz_init(self.ww2, self.quat)
        self.sx2 = nn.Parameter(torch.tensor(0.))
        self.zx2 = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        if 1:    ##如果想用解开方程式的矩阵乘来计算，把三处 if 1: 改成 if self.training:   ,另外不要在这种情况下训练
            if self.sx1 <= 0:
                self.sx1, self.zx1 = sz_init(x, self.quat)  #避免缩放因子为0或负数造成的错误
            x = F.linear(input=trainsz_quantize(x, self.sx1, self.zx1, self.quat),
                         weight=trainsz_quantize(self.ww1, self.sw1, self.zw1, self.quat),
                         bias=self.bb1)
            x = self.net[0](x)
            x = self.net[1](x)
            if self.sx2 <= 0:
                self.sx2, self.zx2 = sz_init(x, self.quat)
            x = F.linear(input=trainsz_quantize(x, self.sx2, self.zx2, self.quat),
                         weight=trainsz_quantize(self.ww2, self.sw2, self.zw2, self.quat),
                         bias=self.bb2)
            x = self.net[2](x)
            return x
        else:
            if self.flag == 0:
                self.w1 = sz_quantize(self.ww1.T, self.sw1, self.zw1, self.quat)
                self.w2 = sz_quantize(self.ww2.T, self.sw2, self.zw2, self.quat)
                self.flag = 1
            x = sz_quantize(x, self.sx1, self.zx1, self.quat)
            x = (torch.matmul(x, self.w1) - self.w1.sum(axis=0, keepdim=True) * torch.round(self.zx1) - x.sum(axis=2,
                                                                                                 keepdim=True) * torch.round(self.zw1) + torch.round(self.zx1) * torch.round(self.zw1) * x.size(
                2)) * self.sx1 * self.sw1 + self.bb1
            x = self.net[0](x)
            x = sz_quantize(x, self.sx2, self.zx2, self.quat)
            x = (torch.matmul(x, self.w2) - self.w2.sum(axis=0, keepdim=True) * torch.round(self.zx2) - x.sum(axis=2,
                                                                                                 keepdim=True) * torch.round(self.zw2) + torch.round(self.zx2) * torch.round(self.zw2) * x.size(
                2)) * self.sx2 * self.sw2 + self.bb2
            return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., quat=4):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.quat = quat
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.flag = 0
        self.attend = nn.Softmax(dim=-1)
        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            #nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.ww1 = nn.Parameter(torch.empty((inner_dim * 3, dim)))
        self.ww2 = nn.Parameter(torch.empty((dim, inner_dim)))
        self.bb2 = nn.Parameter(torch.empty(dim))

        self.sw1, self.zw1 = sz_init(self.ww1, self.quat)
        self.sx1 = nn.Parameter(torch.tensor(0.))
        self.zx1 = nn.Parameter(torch.tensor(0.))
        self.sw2, self.zw2 = sz_init(self.ww2, self.quat)
        self.sx2 = nn.Parameter(torch.tensor(0.))
        self.zx2 = nn.Parameter(torch.tensor(0.))

        self.sq = nn.Parameter(torch.tensor(0.))
        self.zq = nn.Parameter(torch.tensor(0.))
        self.sk = nn.Parameter(torch.tensor(0.))
        self.zk = nn.Parameter(torch.tensor(0.))
        self.sa = nn.Parameter(torch.tensor(0.))
        self.za = nn.Parameter(torch.tensor(0.))
        self.sv = nn.Parameter(torch.tensor(0.))
        self.zv = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        if 1:
            if self.sx1 <= 0:
                self.sx1, self.zx1 = sz_init(x, self.quat)
            qkv = F.linear(input=trainsz_quantize(x, self.sx1, self.zx1, self.quat),
                         weight=trainsz_quantize(self.ww1, self.sw1, self.zw1, self.quat),
                           ).chunk(3,dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
            if self.sq <= 0:
                self.sq, self.zq = sz_init(q, self.quat)
            if self.sk <= 0:
                self.sk, self.zk = sz_init(k, self.quat)
            dots = torch.matmul(trainsz_quantize(q,self.sq,self.zq, self.quat), trainsz_quantize(k,self.sk,self.zk, self.quat).transpose(-1, -2)) * self.scale
            a = self.attend(dots)
            if self.sa <= 0:
                self.sa, self.za = sz_init(a, self.quat)
            if self.sv <= 0:
                self.sv, self.zv = sz_init(v, self.quat)
            out = torch.matmul(trainsz_quantize(a,self.sa,self.za, self.quat), trainsz_quantize(v,self.sv,self.zv, self.quat))
            out = rearrange(out, 'b h n d -> b n (h d)')
            if self.sx2 <= 0:
                self.sx2, self.zx2 = sz_init(out, self.quat)
            out = F.linear(input=trainsz_quantize(out, self.sx2, self.zx2, self.quat),
                         weight=trainsz_quantize(self.ww2, self.sw2, self.zw2, self.quat),
                         bias=self.bb2)
            return self.to_out[0](out)
        else:
            if self.flag == 0:
                self.w1 = sz_quantize(self.ww1.T, self.sw1, self.zw1, self.quat)
                self.w2 = sz_quantize(self.ww2.T, self.sw2, self.zw2, self.quat)
                self.flag = 1
            x = sz_quantize(x, self.sx1, self.zx1, self.quat)
            x = (torch.matmul(x, self.w1) - self.w1.sum(axis=0, keepdim=True) * torch.round(self.zx1) - x.sum(axis=2,
                                                                                           keepdim=True) * torch.round(self.zw1) + torch.round(self.zx1) * torch.round(self.zw1) * x.size(
                2)) * self.sx1 * self.sw1
            qkv = x.chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
            q = sz_quantize(q, self.sq, self.zq, self.quat)
            k = sz_quantize(k.transpose(-1, -2), self.sk, self.zk, self.quat)
            dots = (torch.matmul(q, k) - k.sum(axis=2, keepdim=True) * torch.round(self.zq) - q.sum(axis=3,
                                                                                  keepdim=True) * torch.round(self.zk)) * self.scale * self.sq * self.sk
            a = self.attend(dots)
            a = sz_quantize(a, self.sa, self.za, self.quat)
            v = sz_quantize(v, self.sv, self.zv, self.quat)
            out = (torch.matmul(a, v) - v.sum(axis=2, keepdim=True) * torch.round(self.za) - a.sum(axis=3,
                                                                                 keepdim=True) * torch.round(self.zv) + torch.round(self.za) * torch.round(self.zv) * a.size(
                3)) * self.sa * self.sv
            out = rearrange(out, 'b h n d -> b n (h d)')
            o=sz_quantize(out, self.sx2, self.zx2, self.quat)
            o = (torch.matmul(o, self.w2) - self.w2.sum(axis=0, keepdim=True) * torch.round(self.zx2) - o.sum(axis=2,
                                                                                           keepdim=True) * torch.round(self.zw2) + torch.round(self.zx2) * torch.round(self.zw1) * o.size(
                2)) * self.sx2 * self.sw2+self.bb2
            return o


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,quat=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout,quat=quat)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout,quat=quat))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., quat=4):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height <= 0 and image_width % patch_width <= 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            #nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,quat=quat)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            #nn.Linear(dim, num_classes)
        )
        self.flag = 0
        self.quat = quat

        self.ww1 = nn.Parameter(torch.empty((dim, patch_dim)))
        self.bb1 = nn.Parameter(torch.empty(dim))
        self.ww2 = nn.Parameter(torch.empty((num_classes, dim)))
        self.bb2 = nn.Parameter(torch.empty(num_classes))

        self.sw1, self.zw1 = sz_init(self.ww1, self.quat)
        self.sx1 = nn.Parameter(torch.tensor(0.))
        self.zx1 = nn.Parameter(torch.tensor(0.))
        self.sw2, self.zw2 = sz_init(self.ww2, self.quat)
        self.sx2 = nn.Parameter(torch.tensor(0.))
        self.zx2 = nn.Parameter(torch.tensor(0.))
    def forward(self, img):
        if 1:
            x = self.to_patch_embedding[0](img)
            if self.sx1 <= 0:
                self.sx1, self.zx1 = sz_init(x, self.quat)
            x = F.linear(input=trainsz_quantize(x, self.sx1, self.zx1, self.quat),
                         weight=trainsz_quantize(self.ww1, self.sw1, self.zw1, self.quat),
                         bias=self.bb1)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

            x = self.to_latent(x)
            x = self.mlp_head[0](x)
            if self.sx2 <= 0:
                self.sx2, self.zx2 = sz_init(x, self.quat)
            x = F.linear(input=trainsz_quantize(x, self.sx2, self.zx2, self.quat),
                         weight=trainsz_quantize(self.ww2, self.sw2, self.zw2, self.quat),
                         bias=self.bb2)
        else:
            if self.flag == 0:
                self.w1 = sz_quantize(self.ww1.T, self.sw1, self.zw1, self.quat)
                self.w2 = sz_quantize(self.ww2.T, self.sw2, self.zw2, self.quat)
                self.flag = 1
            x = self.to_patch_embedding[0](img)
            x = sz_quantize(x, self.sx1, self.zx1, self.quat)
            x = (torch.matmul(x, self.w1) - self.w1.sum(axis=0, keepdim=True) * torch.round(self.zx1) - x.sum(axis=2,
                                                                                                 keepdim=True) * torch.round(self.zw1) + torch.round(self.zx1) * torch.round(self.zw1) * x.size(
                2)) * self.sx1 * self.sw1 + self.bb1
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            x = self.mlp_head[0](x)
            x = sz_quantize(x, self.sx2, self.zx2, self.quat)
            x = (torch.matmul(x, self.w2) - self.w2.sum(axis=0, keepdim=True) * torch.round(self.zx2) - x.sum(axis=1,
                                                                                                 keepdim=True) * torch.round(self.zw2) + torch.round(self.zx2) * torch.round(self.zw2) * x.size(
                1)) * self.sx2 * self.sw2 + self.bb2
        return x
