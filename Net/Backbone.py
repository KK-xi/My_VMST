import torch.nn as nn
from Net.Basic_blocks import *
from timm.models.layers import trunc_normal_
from Net.Transformer_blocks import SABlock


class MS_Transformer(nn.Module):
    def __init__(self, num_nodes, in_channels, embed_dim, drop_rate):

        super(MS_Transformer, self).__init__()

        self.num_nodes = num_nodes

        # transformers settings
        self.embed_dim = embed_dim
        drop_rate = drop_rate
        self.in_chans = in_channels

        self.voxel_embed1 = STFE(in_chans=in_channels, embed_dim=self.embed_dim[0])
        self.drop = nn.Dropout(p=drop_rate)

        self.voxel_embed = nn.ModuleList()
        for i in range(len(self.embed_dim) - 1):
            self.voxel_embed.append(
                FPS_VoxelsEmbedding(npoint=num_nodes // 4**(i+1), nsample=16, in_channel=self.embed_dim[i],
                                    out_chan=self.embed_dim[i+1], knn=True))

        self.blocks = nn.ModuleList()
        for i in range(len(self.embed_dim)):
            if i < 2:
                k = [20, 25]
                heads = 2
            elif i == 2:
                k = [20, num_nodes // 4 ** i]
                heads = 2
            else:
                k = [8, num_nodes // 4 ** i]
                heads = 2
            self.blocks.append(SABlock(dim=self.embed_dim[i], num_heads=heads, num_nei=k))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, coords, x):
        """
        :param coords:  shape=[b, N, 3]
        :param x:  shape=[b, N, in_chann, vh, vw]
        :return: preds
        """

        # voxels embedding
        x = self.voxel_embed1(x)
        x = self.drop(x) # b, N, 32
        coords = coords[:, :, 0:3]

        for i in range(len(self.embed_dim)):
            if i != len(self.embed_dim)-1:
                x = self.blocks[i](x, coords)  # b, n, c
                x, coords = self.voxel_embed[i](coords, x)  # b, n, c
            else:
                x = self.blocks[i](x, coords)  # b, n, c

        return x