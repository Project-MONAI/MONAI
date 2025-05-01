import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple placeholder for the SSM (Mamba-like block)
class SSMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, L, C)
        return self.linear2(torch.silu(self.linear1(x)))

class UMambaBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv_res1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(),
        )
        self.conv_res2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(),
        )

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.conv1d = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.ssm = SSMBlock(hidden_channels)

    def forward(self, x):
        # x: (B, C, H, W, D)
        residual = x
        x = self.conv_res1(x)
        x = self.conv_res2(x) + residual

        B, C, H, W, D = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, L, C)
        x_norm = self.layernorm(x_flat)
        x_proj = self.linear1(x_norm)

        x_silu = torch.silu(x_proj)
        x_ssm = self.ssm(x_silu)
        x_conv1d = self.conv1d(x_proj.permute(0, 2, 1)).permute(0, 2, 1)

        x_combined = torch.silu(x_conv1d) * torch.silu(x_ssm)
        x_out = self.linear2(x_combined)
        x_out = x_out.permute(0, 2, 1).view(B, C, H, W, D)

        return x + x_out  # Residual connection

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))

class UMambaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.enc1 = UMambaBlock(in_channels, base_channels)
        self.down1 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1)

        self.enc2 = UMambaBlock(base_channels*2, base_channels*2)
        self.down2 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1)

        self.bottleneck = UMambaBlock(base_channels*4, base_channels*4)

        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels*4)

        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels*2)

        self.final = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        return self.final(x)
