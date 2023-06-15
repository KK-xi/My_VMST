import torch.nn as nn
import torch
from timm.models.layers import DropPath
from einops import rearrange


class MSF(nn.Module):
    def __init__(self, dim, M=2, r=1, act_layer=nn.GELU()):
        """ Constructor
        Args:
            dim: input channel dimensionality.
            M: the number of branchs.
            r: the ratio for compute d, the length of z.
        """
        super().__init__()
        self.dim = dim
        self.channel = dim // M
        assert dim == self.channel * M
        self.d = self.channel // r
        self.M = M
        self.proj = nn.Linear(dim, dim)  # fc

        self.act_layer = act_layer  # fa
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(dim, self.d)  # Ffc + Fa
        self.fc2 = nn.Linear(self.d, self.M * self.channel)
        self.softmax = nn.Softmax(dim=1)
        self.proj_head = nn.Linear(dim, dim)

    def forward(self, input_feats):
        bs, N, _ = input_feats.shape

        input_groups = input_feats.permute(0, 2, 1).contiguous().view(bs, self.M, self.channel, N)

        feats = self.proj(input_feats) # [bs, N, dim]
        feats = self.act_layer(feats)
        feats_proj = feats  # [bs, N, dim]

        feats = feats.permute(0, 2, 1).contiguous()  # [bs, dim, N]
        feats_S = self.gap(feats)  # [bs,dim,1]

        feats_Z = self.act_layer(self.fc1(feats_S.squeeze()))  # [bs,d]

        attention_vectors = self.fc2(feats_Z).view(bs, self.M, self.channel, 1)  # [bs, M, channel, 1]
        attention_vectors = self.softmax(attention_vectors)  # along M axis

        feats_V = input_groups * attention_vectors  # [bs, M, channel, N]
        feats_V = feats_V.view(bs, -1, N).permute(0, 2, 1).contiguous()
        feats_V = self.proj_head(feats_V)  # [bs, N, dim]

        output = feats_proj + feats_V  # [bs,dim, N]

        return output


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm£»
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum(torch.pow((src[:, :, None] - dst[:, None]), 2), dim=-1)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_MSF(nn.Module):
    def __init__(self, dim, num_heads=1, num_nei=None,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_neighbors = num_nei
        self.num_heads = num_heads
        self.g_h = self.num_heads // len(self.num_neighbors) # 1-head for every group
        self.dim = dim
        self.g_dim = dim // len(self.num_neighbors)

        head_dim = self.g_dim // self.g_h # 64
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.pos_mlp_v = nn.ModuleList()
        for i in range(len(self.num_neighbors)):
            self.pos_mlp_v.append(nn.Sequential(
                nn.Linear(3, self.g_dim),
                nn.GELU(),
            ))
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

        if len(self.num_neighbors)>1:
            self.MSF = MSF(dim, len(self.num_neighbors), act_layer=nn.GELU())
        else:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, pos):
        B, N, C = x.shape
        device = x.device
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        dists = square_distance(pos, pos)

        # get queries, keys, values
        qkv_group = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3).contiguous()  # [3, B, N, C]
        qkv_group = qkv_group.chunk(len(self.num_neighbors), -1)

        x_groups = []
        for i in range(len(self.num_neighbors)):
            num_nei = self.num_neighbors[i]
            qkv = qkv_group[i].contiguous()
            q, k, v = qkv[0], qkv[1], qkv[2]
            # split out heads
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.g_h).contiguous()  # b, h, n, d
            if num_nei == N:
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.g_h).contiguous(), (k, v))  # b, h, n, d
                rel_pos = rearrange(pos, 'b i c -> b i 1 c').contiguous() - rearrange(pos,
                                                                                      'b j c -> b 1 j c').contiguous()
                v_rel_pos = self.pos_mlp_v[i](rel_pos)
                v_rel_pos = rearrange(v_rel_pos, 'b n k (h d) -> b h n k d', h=self.g_h).contiguous()  # b, h, n, n, d
                v = v[:, :, None, :, :] + v_rel_pos  # b, h, n, n, d

                attn_rel_pos = v_rel_pos.sum(-1)  # b x n x n
                attn = (q @ k.transpose(-2, -1)).squeeze(-2) * self.scale  # b, h, n, n
                attention = self.softmax(attn + attn_rel_pos)  # b, h, n, n
                attention = rearrange(attention, 'b h i j -> b h i j 1').contiguous()
                x = (attention * v).sum(-2).transpose(1, 2).reshape(B, N, -1)
            else:
                knn_idx = dists.argsort()[:, :, :num_nei]  # b x n x k
                knn_idx = knn_idx.flatten()  # B x npoint x K
                batch_ind = batch_indices.unsqueeze(-1).repeat(1, N * num_nei).flatten()
                knn_xyz = pos[batch_ind, knn_idx, :].view(B, N, num_nei, 3)  # [B, npoint, nsample, 3]
                k = k[batch_ind, knn_idx, :].view(B, N, num_nei, self.g_dim)  # b, n, k, c
                v = v[batch_ind, knn_idx, :].view(B, N, num_nei, self.g_dim)  # b, n, k, c
                k, v = map(lambda t: rearrange(t, 'b n k (h d) -> b h n k d', h=self.g_h).contiguous(),
                           (k, v))  # b, h, n, k, d

                rel_pos = pos[:, :, None] - knn_xyz  # b x n x k x 3
                v_rel_pos = self.pos_mlp_v[i](rel_pos)
                v_rel_pos = rearrange(v_rel_pos, 'b n k (h d) -> b h n k d', h=self.g_h).contiguous()  # b, h, n, k, d
                v = v + v_rel_pos
                attn_rel_pos = v_rel_pos.sum(-1)  # b, h, n, k

                q = q.view(B, self.g_h, N, 1, self.g_dim // self.g_h)
                attn = (q @ k.transpose(-2, -1)).squeeze(-2) * self.scale  # b, h, n, k
                attention = self.softmax(attn + attn_rel_pos)

                attention = rearrange(attention, 'b h i j -> b h i j 1').contiguous()
                x = (attention * v).sum(-2)
                x = x.transpose(1, 2).contiguous().view(B, N, -1)
            x_groups.append(x)

        x = torch.cat(x_groups, -1)

        if len(self.num_neighbors) > 1:
            x = self.MSF(x)
        else:
            x = self.proj(x)
            x = self.proj_drop(x)

        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, num_nei=None, mlp_ratio=1.0,
                 qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention_MSF(dim, num_heads=num_heads, num_nei=num_nei, qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)  # head_split

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, coords):

        x = x + self.drop_path(self.attn(self.norm1(x), coords))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x