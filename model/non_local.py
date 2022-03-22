import torch
from torch import nn
import einops


class NonLocalBlock(nn.Module):
    def __init__(self, channel_feat, channel_inner, gamma=1):
        super().__init__()
        self.q_conv = nn.Conv2d(in_channels=channel_feat,
                                out_channels=channel_inner,
                                kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=channel_feat,
                                out_channels=channel_inner,
                                kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=channel_feat,
                                out_channels=channel_inner,
                                kernel_size=1)
        self.merge_conv = nn.Conv2d(in_channels=channel_inner,
                                    out_channels=channel_feat,
                                    kernel_size=1)
        self.gamma = gamma

    def forward(self, x):
        b, c, h, w = x.shape[:]
        q_tensor = self.q_conv(x)
        k_tensor = self.k_conv(x)
        v_tensor = self.v_conv(x)

        q_tensor = einops.rearrange(q_tensor, 'b c h w -> b c (h w)')
        k_tensor = einops.rearrange(k_tensor, 'b c h w -> b c (h w)')
        v_tensor = einops.rearrange(v_tensor, 'b c h w -> b c (h w)')

        qk_tensor = torch.einsum('b c i, b c j -> b i j', q_tensor, k_tensor)  # where i = j = (h * w)
        attention = torch.softmax(qk_tensor, -1)
        out = torch.einsum('b n i, b c i -> b c n', attention, v_tensor)
        out = einops.rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        out = self.merge_conv(out)
        out = self.gamma * out + x
        return out
