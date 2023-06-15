import torch.nn as nn
from Net.Backbone import *
import torch.nn.functional as F
import torch
from timm.models.layers import trunc_normal_


class VMST_Net(nn.Module):
    def __init__(self, flags):

        super(VMST_Net, self).__init__()

        self.num_classes = flags.num_classes

        self._args = flags
        num_nodes = flags.voxel_num

        # transformer settings
        self.embed_dim = flags.embed_dim
        drop_rate = flags.drop_rate
        in_channels = flags.in_chan
        self.ms_transformer = MS_Transformer(num_nodes=num_nodes, in_channels=in_channels,
                                           embed_dim=self.embed_dim, drop_rate=drop_rate)
        # Classifier head
        self.fc1 = nn.Linear(self.embed_dim[-1]*2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, self.num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, coords, x):
        """
        :param  coords:  shape=[B, N, 3]
        :param x: shape=[B, N, in_chan, vh, vw]
        :return: preds
        """

        x = self.ms_transformer(coords, x) # B, N, C

        x = x.transpose(1, 2).contiguous()

        max_x = F.adaptive_max_pool1d(x, 1).view(-1, self.embed_dim[-1]) # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        avg_x = F.adaptive_avg_pool1d(x, 1).view(-1, self.embed_dim[-1]) # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)

        x = torch.cat([avg_x, max_x], -1)  # B, 1024

        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.fc3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return F.log_softmax(x, dim=1)

