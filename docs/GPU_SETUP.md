# GPU Acceleration Guide

VoxelVR uses ONNX Runtime for 2D pose detection, which can be accelerated with GPU support.

## Checking Your GPU Status

When you run VoxelVR, you'll see which execution providers are being used:

```
Using ONNX Runtime providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

If you see a warning like:
```
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*
```

This means GPU acceleration is not available and the system is falling back to CPU.

## GPU Requirements

| Component | Required Version |
|-----------|------------------|
| CUDA Toolkit | 12.x |
| cuDNN | 9.x |
| NVIDIA Driver | 525+ |

### Supported GPUs

Any NVIDIA GPU with **compute capability 6.0+** is supported, including:
- GeForce GTX 10xx series and newer
- Quadro P-series and newer
- RTX series (all)

## Installation Options

### Option 1: Install CUDA 12 (Recommended)

This gives full GPU acceleration with the latest ONNX Runtime:

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4

# Install cuDNN 9
sudo apt-get install libcudnn9-cuda-12

# Reinstall onnxruntime-gpu
pip install --force-reinstall onnxruntime-gpu
```

After installation, add CUDA to your PATH in `~/.bashrc`:
```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

### Option 2: Use CUDA 11.x Compatible ONNX Runtime

If you have CUDA 11.x installed and don't want to upgrade:

```bash
# Install the last ONNX Runtime version that supports CUDA 11.x
pip install onnxruntime-gpu==1.16.3
```

### Option 3: CPU-Only Mode

No action needed - VoxelVR automatically falls back to CPU execution.
This is slower (~20ms vs ~5ms per frame) but fully functional.

## Verifying GPU Acceleration

Run this command to verify your setup:

```bash
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('Available providers:', providers)
if 'CUDAExecutionProvider' in providers:
    print('✓ GPU acceleration is available')
else:
    print('✗ GPU acceleration NOT available - using CPU')
"
```

## Troubleshooting

### "libcublasLt.so.12: cannot open shared object file"
- CUDA 12.x is not installed or not in your library path
- Solution: Install CUDA 12.x or use Option 2 above

### "Failed to create CUDAExecutionProvider"
- CUDA/cuDNN version mismatch
- Check installed versions: `nvcc --version` and `cat /usr/include/cudnn_version.h`

### GPU shows in nvidia-smi but not used by ONNX Runtime
- Ensure `onnxruntime-gpu` (not just `onnxruntime`) is installed
- Check: `pip show onnxruntime-gpu`

## Performance Comparison

| Mode | Pose Detection Time | Preview FPS (4 cameras) |
|------|---------------------|-------------------------|
| GPU (CUDA 12) | ~5ms | 25-30 FPS |
| CPU (fallback) | ~20ms | 10-15 FPS |
