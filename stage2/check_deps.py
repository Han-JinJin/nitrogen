# -*- coding: utf-8 -*-
"""
Dependency Check Script for NitroGen Stage 2
"""

import os
# 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys

print("=" * 60)
print("NitroGen Stage 2 - Dependency Check")
print("=" * 60)

# Python version
print(f"\nPython version: {sys.version}")

# Check PyTorch
print("\n[1] Checking PyTorch...")
try:
    import torch
    print(f"  [OK] PyTorch {torch.__version__}")
    print(f"  [INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [INFO] GPU: {torch.cuda.get_device_name(0)}")
    TORCH_AVAILABLE = True
except ImportError:
    print("  [FAIL] PyTorch not installed")
    print("  [FIX] pip install torch torchvision")
    TORCH_AVAILABLE = False

# Check transformers
print("\n[2] Checking Transformers...")
try:
    import transformers
    print(f"  [OK] Transformers {transformers.__version__}")
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("  [WARN] Transformers not installed (optional)")
    print("  [FIX] pip install transformers")
    TRANSFORMERS_AVAILABLE = False

# Check OpenCV
print("\n[3] Checking OpenCV...")
try:
    import cv2
    print(f"  [OK] OpenCV {cv2.__version__}")
    CV2_AVAILABLE = True
except ImportError:
    print("  [FAIL] OpenCV not installed")
    print("  [FIX] pip install opencv-python")
    CV2_AVAILABLE = False

# Check NumPy
print("\n[4] Checking NumPy...")
try:
    import numpy as np
    print(f"  [OK] NumPy {np.__version__}")
    NUMPY_AVAILABLE = True
except ImportError:
    print("  [FAIL] NumPy not installed")
    print("  [FIX] pip install numpy")
    NUMPY_AVAILABLE = False

# Test models
print("\n" + "=" * 60)
print("Model Testing")
print("=" * 60)

if not TORCH_AVAILABLE:
    print("\n[SKIP] Cannot test models - PyTorch not installed")
else:
    print("\n[A] Testing SegFormer Model...")
    try:
        from models import NitroGenActionParser

        model = NitroGenActionParser(
            num_buttons=16,
            use_pretrained=True  # 使用默认的本地模型路径
        )
        frame_pairs = torch.randn(1, 6, 256, 256)
        outputs = model(frame_pairs)

        print("  [OK] SegFormer model loaded from local files")
        print("  [INFO] Output shapes:")
        for k, v in outputs.items():
            print(f"    - {k}: {v.shape}")

    except Exception as e:
        print(f"  [FAIL] SegFormer model test failed: {e}")
        if not TRANSFORMERS_AVAILABLE:
            print("  [INFO] Install transformers: pip install transformers")
        else:
            print("  [INFO] Ensure segformer-b5-local/ directory contains:")
            print("    - config.json")
            print("    - pytorch_model.bin (or model.safetensors)")
            print("    - preprocessor_config.json")

# Summary
print("\n" + "=" * 60)
if all([TORCH_AVAILABLE, CV2_AVAILABLE, NUMPY_AVAILABLE]):
    print("Summary & Recommendations")
    print("=" * 60)

    print("\n✅ All core dependencies are installed!")
    print(f"  • PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    print(f"  • OpenCV {cv2.__version__}")
    print(f"  • NumPy {np.__version__}")

    if TRANSFORMERS_AVAILABLE:
        print(f"  • Transformers {transformers.__version__}")

    print("\n🎉 You're ready to use NitroGen Stage 2!")

else:
    print("Missing Dependencies")
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch (required)")
        print("   pip install torch torchvision")

    if not CV2_AVAILABLE:
        print("\n❌ OpenCV (required)")
        print("   pip install opencv-python")

    if not NUMPY_AVAILABLE:
        print("\n❌ NumPy (required)")
        print("   pip install numpy")

    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  Transformers (optional, for SegFormer)")
        print("   pip install transformers")

print("\n" + "=" * 60)
