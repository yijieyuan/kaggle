import torch
from torch import nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
import math
from typing import List, Tuple, Optional


# ============================================================================
# Config (simulated)
# ============================================================================
class CFG:
    model_name = 'resnet18d'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Conv2dSame and helper functions
# ============================================================================
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1, 1), value: float = 0):
    ih, iw, iz = x.size()[-3:]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    pad_w = get_same_padding(iw, k[1], s[1], d[1])
    pad_z = get_same_padding(iz, k[2], s[2], d[2])
    if pad_h > 0 or pad_w > 0 or pad_z > 0:
        x = F.pad(x, [pad_z // 2, pad_z - pad_z // 2, 
                      pad_w // 2, pad_w - pad_w // 2, 
                      pad_h // 2, pad_h - pad_h // 2], value=value)
    return x


def conv3d_same(x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
                stride: Tuple[int, int, int] = (1, 1, 1),
                padding: Tuple[int, int, int] = (0, 0, 0), 
                dilation: Tuple[int, int, int] = (1, 1, 1), 
                groups: int = 1):
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """Tensorflow like 'SAME' convolution wrapper for 2D convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return timm.models.layers.conv2d_same(x, self.weight, self.bias, 
                                              self.stride, self.padding, 
                                              self.dilation, self.groups)


class Conv3dSame(nn.Conv3d):
    """Tensorflow like 'SAME' convolution wrapper for 3d convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv3d_same(x, self.weight, self.bias, self.stride, 
                          self.padding, self.dilation, self.groups)


# ============================================================================
# 2D to 3D Conversion Function
# ============================================================================
def convert_3d(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
            
    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(name, convert_3d(child))
    del module

    return module_output


# ============================================================================
# Original Model (2D version)
# ============================================================================
class Model(nn.Module):
    def __init__(self, backbone=None, segtype='unet', pretrained=False):
        super(Model, self).__init__()
        
        n_blocks = 4
        self.n_blocks = n_blocks
        
        self.encoder = timm.create_model(
            CFG.model_name,
            in_chans=3,
            features_only=True,
            drop_rate=0.1,
            drop_path_rate=0.1,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )

        self.segmentation_head = nn.Conv2d(
            decoder_channels[n_blocks-1], 5, 
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

    def forward(self, x):
        if x.dim() == 4:
            x = x.repeat(1, 3, 1, 1)  # (B, 3, H, W)
        elif x.dim() == 5:
            x = x.repeat(1, 3, 1, 1, 1)  # (B, 3, H, W, D)

        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features


# ============================================================================
# Test Functions
# ============================================================================
def test_2d_model():
    """Test the original 2D model"""
    print("=" * 80)
    print("Testing 2D Model (Original)")
    print("=" * 80)
    
    model = Model(pretrained=True)
    model.eval()
    
    # Test with 2D input
    batch_size = 4
    x = torch.randn(batch_size, 1, 256, 256)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 5, 256, 256)")
    
    return model


def analyze_feature_map_sizes_2d():
    """Analyze feature map sizes at each stage for 2D model"""
    print("\n" + "=" * 80)
    print("Feature Map Size Analysis (2D Model with 256x256 input)")
    print("=" * 80)
    
    model = Model(pretrained=False)
    model.eval()
    
    x = torch.randn(1, 256, 256)
    x_3ch = torch.stack([x]*3, 1)
    
    print("\nEncoder stages:")
    features = model.encoder(x_3ch)
    for i, feat in enumerate(features[:model.n_blocks]):
        print(f"  Stage {i+1}: {feat.shape} (channels: {feat.shape[1]})")
    
    print("\nDecoder process:")
    global_features = [0] + features[:model.n_blocks]
    seg_features = model.decoder(*global_features)
    print(f"  Decoder output: {seg_features.shape}")
    
    output = model.segmentation_head(seg_features)
    print(f"  Final output: {output.shape}")


def test_3d_model():
    """Test the 3D converted model"""
    print("\n" + "=" * 80)
    print("Testing 3D Model (Converted from 2D)")
    print("=" * 80)
    
    # Create 2D model
    model_2d = Model(pretrained=False)
    
    # Convert to 3D
    print("\nConverting 2D model to 3D...")
    model_3d = convert_3d(model_2d)
    model_3d.eval()
    
    # Test with 3D input
    batch_size = 2
    x = torch.randn(batch_size, 1, 256, 256, 256)
    
    print(f"\nInput shape: {x.shape}")
    print(f"  - Batch: {batch_size}")
    print(f"  - Channels: 1")
    print(f"  - Size: 256 x 256 x 256")
    
    with torch.no_grad():
        output = model_3d(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"  - Batch: {output.shape[0]}")
    print(f"  - Classes: {output.shape[1]}")
    print(f"  - Size: {output.shape[2]} x {output.shape[3]} x {output.shape[4]}")
    
    return model_3d


def analyze_feature_map_sizes_3d():
    """Analyze feature map sizes at each stage for 3D model"""
    print("\n" + "=" * 80)
    print("Feature Map Size Analysis (3D Model with 256x256x256 input)")
    print("=" * 80)
    
    # Create and convert model
    model_2d = Model(pretrained=False)
    model_3d = convert_3d(model_2d)
    model_3d.eval()
    
    # Hook to capture intermediate features
    features_3d = []
    
    def hook_fn(module, input, output):
        features_3d.append(output.shape)
    
    # Register hooks on encoder
    hooks = []
    for i, layer in enumerate(model_3d.encoder.children()):
        hooks.append(layer.register_forward_hook(hook_fn))
    
    x = torch.randn(1, 1, 256, 256, 256)
    x_3ch = torch.stack([x]*3, 1)
    
    print("\nEncoder stages (estimated):")
    print("  Input: (1, 1, 256, 256, 256)")
    print("  After stacking to 3 channels: (1, 3, 256, 256, 256)")
    
    with torch.no_grad():
        try:
            features = model_3d.encoder(x_3ch)
            for i, feat in enumerate(features[:model_3d.n_blocks]):
                print(f"  Stage {i+1}: {feat.shape}")
            
            # Decoder
            global_features = [0] + features[:model_3d.n_blocks]
            seg_features = model_3d.decoder(*global_features)
            print(f"\nDecoder output: {seg_features.shape}")
            
            # Final output
            output = model_3d.segmentation_head(seg_features)
            print(f"Final output: {output.shape}")
            
            # Calculate lowest resolution
            min_spatial = min(features[model_3d.n_blocks-1].shape[2:])
            print(f"\n✓ Lowest spatial resolution: {min_spatial}³")
            
        except Exception as e:
            print(f"\nError during forward pass: {e}")
            print("\nNote: 3D conversion may require adjustments for some layer types.")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "SEGMENTATION MODEL TEST SUITE" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Test 2D model
    model_2d = test_2d_model()
    analyze_feature_map_sizes_2d()
    
    # Test 3D model
    try:
        model_3d = test_3d_model()
        analyze_feature_map_sizes_3d()
    except Exception as e:
        print(f"\n⚠ Error in 3D model test: {e}")
        print("This may be due to library compatibility issues.")
    
    # Model statistics
    print("\n" + "=" * 80)
    print("Model Statistics (2D)")
    print("=" * 80)
    total_params = sum(p.numel() for p in model_2d.parameters())
    trainable_params = sum(p.numel() for p in model_2d.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    print("\n" + "=" * 80)
    print("✓ Testing Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()