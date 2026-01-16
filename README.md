# VoxelVR - Multi-Camera Full Body Tracking for VRChat

A trackerless full body tracking solution using 3+ webcams and volumetric 3D pose estimation.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run calibration wizard
python run_calibration.py

# Run demo visualizer (no VRChat needed)
python run_demo.py

# Run full tracking pipeline
python run_tracking.py
```

## Requirements

- Python 3.10+
- 3+ USB webcams
- GPU with DirectML/CUDA/ROCm support
- ChArUco calibration board (printable PDF included)

## Project Structure

```
voxelvr/
├── calibration/    # Camera intrinsics/extrinsics
├── capture/        # Multi-camera capture pipeline
├── pose/           # 2D detection + 3D fusion
├── transport/      # VRChat OSC output
├── demo/           # Standalone visualizer
└── gui/            # User interface
```

## License

MIT
