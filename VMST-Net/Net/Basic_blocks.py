import torch.nn as nn
import torch
import torch.nn.functional as F

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

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum(torch.pow((xyz-centroid), 2), -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    device = points.device
    B, N, C = points.shape
    S = npoint
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    fps_idx = farthest_point_sample(xyz, S)  # [B, npoint]
    batch_ind = batch_indices.unsqueeze(-1).repeat(1, S).flatten()
    new_xyz = xyz[batch_ind, fps_idx.flatten(), :].view(B, S, 3)
    new_x = points[batch_ind, fps_idx.flatten(), :].view(B, S, C)

    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)

    idx = idx.flatten()  # B x npoint x K
    batch_ind = batch_indices.unsqueeze(-1).repeat(1, S * nsample).flatten()

    grouped_xyz = xyz[batch_ind, idx, :].view(B, S, nsample, 3) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)

    grouped_points = points[batch_ind, idx, :].view(B, S, nsample, C)
    new_x = new_x.unsqueeze(2).repeat(1, 1, nsample, 1)
    feature = torch.cat((grouped_points - new_x, grouped_xyz, grouped_points), dim=3).permute(0, 3, 2, 1).contiguous()
    new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]

    return new_xyz, new_points, feature


class FPS_VoxelsEmbedding(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_chan, knn=False, radius=0):
        super(FPS_VoxelsEmbedding, self).__init__()
        self.in_chan = in_channel
        self.out_chan = out_chan
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        last_channel = in_channel + 3
        self.mlp_convs = nn.Conv2d(last_channel, self.out_chan, 1)
        self.mlp_ln = nn.LayerNorm(self.out_chan)

    def forward(self, xyz, voxels):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            voxels: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, 3]
            new_voxels: sample points feature data, [B, S, C_out]
        """

        new_xyz, new_points, feature = sample_and_group(self.npoint, self.radius, self.nsample, xyz, voxels, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_voxels: sampled voxels data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1).contiguous() # [B, C+D, nsample,npoint]
        new_points = self.mlp_convs(new_points)
        new_points = torch.max(new_points, 2)[0].transpose(1, 2).contiguous()

        new_points = self.mlp_ln(new_points)

        return new_points, new_xyz


class STFE(nn.Module):
    """ Image to Patch Embedding
    in_chans = 2
    head_embed = conv
    data = time
    """
    def __init__(self, in_chans=1, embed_dim=32):
        super(STFE, self).__init__()
        self.in_chan = in_chans

        self.dim = embed_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, self.dim//2, 5, 1, 2),
            nn.BatchNorm2d(self.dim//2),
            nn.GELU(),
            nn.AdaptiveMaxPool2d((5, 5)),
            nn.Conv2d(self.dim // 2, self.dim, 3, 1, 1),
            nn.BatchNorm2d(self.dim),
            nn.GELU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )

        self.ln = nn.LayerNorm(self.dim)

    def forward(self, x):
        """
        :param x: shape=[B, N, 10, 10] /[B, N, 5, 5]
        :param mask:
        :return:
        """

        if self.in_chan == 1:
            B, N, H, W = x.shape
            x = x.view(B * N, 1, H, W)
        else:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)

        x = self.conv1(x)
        x = self.ln(x.view(B, N, -1))

        return x
