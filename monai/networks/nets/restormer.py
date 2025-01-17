import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))




import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.upsample import UpSample, UpsampleMode
from monai.networks.blocks.downsample import DownSample, DownsampleMode
from monai.networks.layers.factories import Norm
from einops import rearrange

class FeedForward(nn.Module):
    """Gated-DConv Feed-Forward Network (GDFN) that controls feature flow using gating mechanism.
    Uses depth-wise convolutions for local context mixing and GELU-activated gating for refined feature selection."""
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, 
                               stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA) Differs from standard self-attention
    by operating on feature channels instead of spatial dimensions. Incorporates depth-wise
    convolutions for local mixing before attention, achieving linear complexity vs quadratic
    in vanilla attention."""
    def __init__(self, dim: int, num_heads: int, bias: bool, flash_attention: bool = False):
        super().__init__()
        if flash_attention and not hasattr(F, 'scaled_dot_product_attention'):
            raise ValueError("Flash attention not available")
            
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.flash_attention = flash_attention
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, 
                                   padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self._attention_fn = self._get_attention_fn()

    def _get_attention_fn(self):
        if self.flash_attention:
            return self._flash_attention
        return self._normal_attention
    def _flash_attention(self, q, k, v):
        """Flash attention implementation using scaled dot-product attention."""
        scale = float(self.temperature.mean())  
        out = F.scaled_dot_product_attention(
            q,
            k, 
            v,
            scale=scale,
            dropout_p=0.0,
            is_causal=False
        )
        return out

    def _normal_attention(self, q, k, v):
        """Attention matrix multiplication with depth-wise convolutions."""
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        return attn @ v
    def forward(self, x):
        """Forward pass for MDTA attention. 
        1. Apply depth-wise convolutions to Q, K, V
        2. Reshape Q, K, V for multi-head attention
        3. Compute attention matrix using flash or normal attention
        4. Reshape and project out attention output"""
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        out = self._attention_fn(q, k, v)        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    """Basic transformer unit combining MDTA and GDFN with skip connections.
    Unlike standard transformers that use LayerNorm, this block uses Instance Norm
    for better adaptation to image restoration tasks."""
    
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float,
                 bias: bool, LayerNorm_type: str, flash_attention: bool = False):
        super().__init__()
        use_bias = LayerNorm_type != 'BiasFree'        
        self.norm1 = Norm[Norm.INSTANCE, 2](dim, affine=use_bias)
        self.attn = Attention(dim, num_heads, bias, flash_attention)
        self.norm2 = Norm[Norm.INSTANCE, 2](dim, affine=use_bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(f'x shape in transformer block: {x.shape}')

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Initial feature extraction using overlapped convolutions.
    Unlike standard patch embeddings that use non-overlapping patches,
    this approach maintains spatial continuity through 3x3 convolutions."""
    
    def __init__(self, in_c: int = 3, embed_dim: int = 48, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, 
                             stride=1, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)



class DownSample_local(nn.Module):
    """Downsampling module that halves spatial dimensions while doubling channels.
    Uses PixelUnshuffle for efficient feature map manipulation."""
    
    def __init__(self, n_feat: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)



class Restormer_new(nn.Module):
    """Restormer: Efficient Transformer for High-Resolution Image Restoration.
    
    Implements a U-Net style architecture with transformer blocks, combining:
    - Multi-scale feature processing through progressive down/upsampling
    - Efficient attention via MDTA blocks
    - Local feature mixing through GDFN
    - Skip connections for preserving spatial details
    
    Architecture:
        - Encoder: Progressive feature downsampling with increasing channels
        - Latent: Deep feature processing at lowest resolution
        - Decoder: Progressive upsampling with skip connections
        - Refinement: Final feature enhancement
    """
    def __init__(self, 
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 1, 1, 1],  
                 heads=[1, 1, 1, 1],  
                 num_refinement_blocks=4,    
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False,
                 flash_attention=False):
        super().__init__()
        """Initialize Restormer model.
        
        Args:
            inp_channels: Number of input image channels
            out_channels: Number of output image channels
            dim: Base feature dimension
            num_blocks: Number of transformer blocks at each scale
            num_refinement_blocks: Number of final refinement blocks
            heads: Number of attention heads at each scale
            ffn_expansion_factor: Expansion factor for feed-forward network
            bias: Whether to use bias in convolutions
            LayerNorm_type: Type of normalization ('WithBias' or 'BiasFree')
            dual_pixel_task: Enable dual-pixel specific processing
            flash_attention: Use flash attention if available
        """
        # Check input parameters
        assert len(num_blocks) > 1, "Number of blocks must be greater than 1"
        assert len(num_blocks) == len(heads), "Number of blocks and heads must be equal"
        assert all([n > 0 for n in num_blocks]), "Number of blocks must be greater than 0"
        
        # Initial feature extraction
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()
        num_steps = len(num_blocks) - 1 
        self.num_steps = num_steps

        # Define encoder levels
        for n in range(num_steps):
            current_dim = dim * 2**n
            self.encoder_levels.append(
                nn.Sequential(*[
                    TransformerBlock(
                        dim=current_dim,
                        num_heads=heads[n],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        flash_attention=flash_attention
                    ) for _ in range(num_blocks[n])
                ])
            )
            print(f' Encoder layer {n}')
            print(f'input channels to the downsampler: {current_dim}')
            print(f'output channels from the downsampler: {current_dim//2}')
            self.downsamples.append(
                #DownSample_local(current_dim)
                DownSample(
                spatial_dims=2,
                in_channels=current_dim,
                out_channels=current_dim//2,
                mode=DownsampleMode.PIXELUNSHUFFLE,
                scale_factor=2,
                bias=bias,
                )
            )

        # Define latent space
        latent_dim = dim * 2**num_steps
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=latent_dim,
                num_heads=heads[num_steps],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                flash_attention=flash_attention
            ) for _ in range(num_blocks[num_steps])
        ])

        # Define decoder levels
        for n in reversed(range(num_steps)):
            current_dim = dim * 2**n
            next_dim = dim * 2**(n+1)
            self.upsamples.append(
                UpSample(
                spatial_dims=2,
                in_channels=next_dim,
                out_channels=(next_dim//2),
                mode=UpsampleMode.PIXELSHUFFLE,
                scale_factor=2,
                bias=bias,
                apply_pad_pool=False
            )
                )
            
            # Reduce channel layers to deal with skip connections
            if n != 0:
                self.reduce_channels.append(
                    nn.Conv2d(next_dim, current_dim, kernel_size=1, bias=bias)
                    )
                decoder_dim = current_dim
            else:
                decoder_dim = next_dim
            
            self.decoder_levels.append(
                nn.Sequential(*[
                    TransformerBlock(
                        dim=decoder_dim,
                        num_heads=heads[n],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        flash_attention=flash_attention
                    ) for _ in range(num_blocks[n])
                ])
            )

        # Final refinement and output
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=decoder_dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                flash_attention=flash_attention
            ) for _ in range(num_refinement_blocks)
        ])
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        #print(f'downsample layer in new model: {self.downsamples}')
        print(f'======================')
        print(f'======================')
        #print(f'upsamples layer in new model: {self.upsamples}')

    def forward(self, x):
        """Forward pass of Restormer.
        Processes input through encoder-decoder architecture with skip connections.
        Args:
            inp_img: Input image tensor of shape (B, C, H, W)
            
        Returns:
            Restored image tensor of shape (B, C, H, W)
        """
        assert x.shape[-1] > 2 ** self.num_steps and x.shape[-2] > 2 ** self.num_steps, "Input dimensions should be larger than 2^number_of_step"

        # Patch embedding
        x = self.patch_embed(x)
        skip_connections = []

        # Encoding path
        for idx, (encoder, downsample) in enumerate(zip(self.encoder_levels, self.downsamples)):
            print(f'image shape at input: {x.shape}')
            x = encoder(x)
            skip_connections.append(x)
            print(f'x shape in new model encoder: {x.shape}')
            x = downsample(x)
            print(f'x shape in new model downsample: {x.shape}')

        # Latent space
        x = self.latent(x)
        
        # Decoding path
        for idx in range(len(self.decoder_levels)):
            x = self.upsamples[idx](x)     
            x = torch.concat([x, skip_connections[-(idx + 1)]], 1)
            if idx < len(self.decoder_levels) - 1:
                x = self.reduce_channels[idx](x)                
            x = self.decoder_levels[idx](x)
                
        # Final refinement
        x = self.refinement(x)

        if self.dual_pixel_task:
            x = x + self.skip_conv(skip_connections[0])
            x = self.output(x)
        else:
            x = self.output(x)

        return x



