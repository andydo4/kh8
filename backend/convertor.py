#!/usr/bin/env python3
"""
Create a simple FSRCNN model in PyTorch and export to ONNX
This bypasses the problematic TensorFlow conversion
"""

import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not found. Install with:")
    print("  pip install torch torchvision")
    sys.exit(1)


class FSRCNN(nn.Module):
    """FSRCNN model for 2x super resolution"""

    def __init__(self, scale_factor=2, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()

        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU()
        )

        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU()
        )

        # Mapping (m layers)
        mapping_layers = []
        for _ in range(m):
            mapping_layers.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU()
            ])
        self.mapping = nn.Sequential(*mapping_layers)

        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU()
        )

        # Deconvolution
        self.deconv = nn.ConvTranspose2d(
            d, num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=4,
            output_padding=scale_factor - 1
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x


def main():
    print("Creating FSRCNN PyTorch model...")
    model = FSRCNN(scale_factor=2)
    model.eval()

    # Create dummy input (channels-first: batch_size=1, channels=3, height=480, width=854)
    dummy_input = torch.randn(1, 3, 480, 854)

    print("Testing model output shape...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    print("\nExporting to ONNX...")
    output_file = "FSRCNN_x2_pytorch.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print(f"✓ Model exported to: {output_file}")
    print("\n" + "=" * 60)
    print("IMPORTANT NOTE:")
    print("=" * 60)
    print("This is an UNTRAINED model with random weights!")
    print("It will upscale images but quality will be poor (blurry).")
    print("\nFor testing ROCm performance, this is fine.")
    print("For production quality, you need pretrained weights.")
    print("=" * 60)


if __name__ == '__main__':
    main()