import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from time import time


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def ste_round(x):
    return torch.round(x) - x.detach() + x


def quantize(q, quat):
    sq = (q.max() - q.min()) / (2 ** quat - 1)
    zq = torch.round(2 ** quat - 1 - q.max() / sq)
    q = torch.round(q / sq + zq)
    return q, nn.Parameter(sq), nn.Parameter(zq)


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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

    def trans(self):
        self.fn.trans()

    def delet(self):
        self.fn.delet()


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., quat=4, ifq=[1, 1]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.flag = 0
        self.quat = quat
        self.ifq = ifq

    def forward(self, x):
        if self.training:
            return self.net(x)
        else:
            if self.ifq[0] != 0:
                x, self.sx1, self.zx1 = quantize(x, self.ifq[0])
                self.w1, self.sw1, self.zw1 = quantize(self.net[0].weight.data.T, self.ifq[0])
                x = (torch.matmul(x, self.w1) - self.w1.sum(axis=0, keepdim=True) * self.zx1 - x.sum(axis=2,
                                                                                                     keepdim=True) * self.zw1 + self.zx1 * self.zw1 * x.size(
                    2)) * self.sx1 * self.sw1 + self.net[0].bias.data
            else:
                x = self.net[0](x)

            x = self.net[1](x)

            if self.ifq[1] != 0:
                x, self.sx2, self.zx2 = quantize(x, self.ifq[1])
                self.w2, self.sw2, self.zw2 = quantize(self.net[3].weight.data.T, self.ifq[1])
                x = (torch.matmul(x, self.w2) - self.w2.sum(axis=0, keepdim=True) * self.zx2 - x.sum(axis=2,
                                                                                                     keepdim=True) * self.zw2 + self.zx2 * self.zw2 * x.size(
                    2)) * self.sx2 * self.sw2 + self.net[3].bias.data
            else:
                x = self.net[3](x)

            return x

    def trans(self):
        self.sw1, self.zw1 = sz_init(self.net[0].weight.data, self.quat)
        self.sx1 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zx1 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.sw2, self.zw2 = sz_init(self.net[3].weight.data, self.quat)
        self.sx2 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zx2 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.ww1 = nn.Parameter(self.net[0].weight.data.clone())
        self.ww2 = nn.Parameter(self.net[3].weight.data.clone())
        self.bb1 = nn.Parameter(self.net[0].bias.data.clone())
        self.bb2 = nn.Parameter(self.net[3].bias.data.clone())

    def delet(self):
        del self.net[0]
        del self.net[2]


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., quat=4, ifq=[1, 1, 1, 1]):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.quat = quat
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.flag = 0
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.ifq = ifq

    def forward(self, x):
        if self.training:
            # if 1:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)
        else:
            if self.ifq[0] != 0:
                x, self.sx1, self.zx1 = quantize(x, self.ifq[0])
                self.w1, self.sw1, self.zw1 = quantize(self.to_qkv.weight.data.T, self.ifq[0])
                x = (torch.matmul(x, self.w1) - self.w1.sum(axis=0, keepdim=True) * self.zx1 - x.sum(axis=2,
                                                                                                     keepdim=True) * self.zw1 + self.zx1 * self.zw1 * x.size(
                    2)) * self.sx1 * self.sw1
            else:
                x = self.to_qkv(x)
            qkv = x.chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
            if self.ifq[1] != 0:
                q, self.sq, self.zq = quantize(q, self.ifq[1])
                k, self.sk, self.zk = quantize(k.transpose(-1, -2), self.ifq[1])
                dots = (torch.matmul(q, k) - k.sum(axis=2, keepdim=True) * self.zq - q.sum(axis=3,
                                                                                           keepdim=True) * self.zk) * self.scale * self.sq * self.sk
            else:
                dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            a = self.attend(dots)
            if self.ifq[2] != 0:
                a, self.sa, self.za = quantize(a, self.ifq[2])
                v, self.sv, self.zv = quantize(v, self.ifq[2])
                out = (torch.matmul(a, v) - v.sum(axis=2, keepdim=True) * self.za - a.sum(axis=3,
                                                                                          keepdim=True) * self.zv + self.za * self.zv * a.size(
                    3)) * self.sa * self.sv
            else:
                out = torch.matmul(a, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            if self.ifq[3] != 0:
                o, self.sx2, self.zx2 = quantize(out, self.ifq[3])
                self.w2, self.sw2, self.zw2 = quantize(self.to_out[0].weight.data.T, self.ifq[3])
                o = (torch.matmul(o, self.w2) - self.w2.sum(axis=0, keepdim=True) * self.zx2 - x.sum(axis=2,
                                                                                                     keepdim=True) * self.zw2 + self.zx2 * self.zw2 * x.size(
                    2)) * self.sx2 * self.sw2 + self.to_out[0].bias.data
            else:
                o = self.to_out[0](out)
            return o

    def trans(self):
        self.sw1, self.zw1 = sz_init(self.to_qkv.weight.data, self.quat)
        self.sx1 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zx1 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.sw2, self.zw2 = sz_init(self.to_out[0].weight.data, self.quat)
        self.sx2 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zx2 = nn.Parameter(torch.tensor(0.).to('cuda'))

        self.sq = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zq = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.sk = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zk = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.sa = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.za = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.sv = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zv = nn.Parameter(torch.tensor(0.).to('cuda'))

        self.ww1 = nn.Parameter(self.to_qkv.weight.data.clone())
        self.ww2 = nn.Parameter(self.to_out[0].weight.data.clone())
        self.bb2 = nn.Parameter(self.to_out[0].bias.data.clone())

    def delet(self):
        del self.to_qkv
        del self.to_out[0]


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,
                 ifq=[1] * 36):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, ifq=ifq[i * 6:4 + i * 6])),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout, ifq=ifq[4 + i * 6:6 + i * 6]))
            ]))
        self.ifq = ifq
        self.depth = depth

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def trans(self):
        for p1, p2 in self.layers:
            p1.trans()
            p2.trans()

    def delet(self):
        for p1, p2 in self.layers:
            p1.delet()
            p2.delet()

    def updat(self, ifq):
        for i in range(self.depth):
            self.layers[i][0].fn.ifq = ifq[i * 6:4 + i * 6]
            self.layers[i][1].fn.ifq = ifq[4 + i * 6:6 + i * 6]


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., quat=4,
                 ifq=[1] * 38):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, ifq[1:37])

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.flag = 0
        self.quat = quat
        self.ifq = ifq

    def forward(self, img):
        if self.training:
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

            x = self.to_latent(x)
            x = self.mlp_head(x)
        else:
            x = self.to_patch_embedding[0](img)
            if self.ifq[0] != 0:
                x, self.sx1, self.zx1 = quantize(x, self.ifq[0])
                self.w1, self.sw1, self.zw1 = quantize(self.to_patch_embedding[1].weight.data.T, self.ifq[0])
                x = (torch.matmul(x, self.w1) - self.w1.sum(axis=0, keepdim=True) * self.zx1 - x.sum(axis=2,
                                                                                                     keepdim=True) * self.zw1 + self.zx1 * self.zw1 * x.size(
                    2)) * self.sx1 * self.sw1 + self.to_patch_embedding[1].bias.data
            else:
                x = self.to_patch_embedding[1](x)
            b, n, _ = x.shape
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            x = self.mlp_head[0](x)
            if self.ifq[37] != 0:
                x, self.sx2, self.zx2 = quantize(x, self.ifq[37])
                self.w2, self.sw2, self.zw2 = quantize(self.mlp_head[1].weight.data.T, self.ifq[37])
                x = (torch.matmul(x, self.w2) - self.w2.sum(axis=0, keepdim=True) * self.zx2 - x.sum(axis=1,
                                                                                                     keepdim=True) * self.zw2 + self.zx2 * self.zw2 * x.size(
                    1)) * self.sx2 * self.sw2 + self.mlp_head[1].bias.data
            else:
                x = self.mlp_head[1](x)
        return x

    def trans(self):
        self.sw1, self.zw1 = sz_init(self.to_patch_embedding[1].weight.data, self.quat)
        self.sx1 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zx1 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.sw2, self.zw2 = sz_init(self.mlp_head[1].weight.data, self.quat)
        self.sx2 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.zx2 = nn.Parameter(torch.tensor(0.).to('cuda'))
        self.ww1 = nn.Parameter(self.to_patch_embedding[1].weight.data.clone())
        self.ww2 = nn.Parameter(self.mlp_head[1].weight.data.clone())
        self.bb1 = nn.Parameter(self.to_patch_embedding[1].bias.data.clone())
        self.bb2 = nn.Parameter(self.mlp_head[1].bias.data.clone())
        self.transformer.trans()

    def delet(self):
        del self.to_patch_embedding[1]
        del self.mlp_head[1]
        self.transformer.delet()

    def updat(self, ifq):
        self.ifq = ifq
        self.transformer.updat(ifq[1:37])
