# VoxelVR - Multi-Camera Full Body Tracking for VRChat

A trackerless full body tracking solution using 3+ webcams and volumetric 3D pose estimation.

## Quick Start

### GUI Mode (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the GUI
python run_gui.py
```

The GUI provides:
- **Camera Preview** - View all camera feeds for positioning
- **Calibration Wizard** - Step-by-step ChArUco board calibration
- **Tracking Controls** - Start/stop tracking with OSC configuration
- **Performance Monitor** - Real-time FPS and latency graphs
- **Debug Panel** - Filter parameter tuning with auto-adjustment

### CLI Mode

```bash
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
- ChArUco calibration board (printable as PDF from GUI or CLI)
- A4 or Letter paper for printing the board (default 5x5 is sized for standard paper)

## GUI Features

### Camera Preview
View all connected cameras in a flexible grid layout. Useful for positioning cameras to cover your play area.

### Calibration Wizard
Step-by-step guidance for calibrating your camera system:
1. Export and print the ChArUco calibration board (PDF format)
2. Capture intrinsic frames for each camera
3. Capture extrinsic frames with all cameras seeing the board
4. Review calibration quality metrics

### Debug & Tuning
Fine-tune filter parameters to balance between jitter and latency:
- **Low Jitter** - Maximum smoothing for minimal shake
- **Balanced** - Good compromise for general use
- **Low Latency** - Fastest response for fast movements
- **Precision** - Optimized for slow, precise movements

The auto-adjustment system continuously optimizes parameters based on your movement patterns.

## Project Structure

```
voxelvr/
├── calibration/    # Camera intrinsics/extrinsics
├── capture/        # Multi-camera capture pipeline
├── pose/           # 2D detection + 3D fusion
├── transport/      # VRChat OSC output
├── demo/           # Standalone visualizer
└── gui/            # Graphical user interface
    ├── app.py              # Main application
    ├── camera_panel.py     # Camera preview
    ├── calibration_panel.py # Calibration wizard
    ├── tracking_panel.py   # Tracking controls
    ├── performance_panel.py # Performance monitor
    ├── debug_panel.py      # Debug & tuning
    ├── osc_status.py       # OSC status indicator
    └── param_optimizer.py  # Auto-adjustment engine
```

## Performance Profiles

| Profile | Use Case | Jitter | Latency |
|---------|----------|--------|---------|
| Low Jitter | Dancing, presentations | Very Low | ~100ms |
| Balanced | General VR | Low | ~50ms |
| Low Latency | Gaming, fast movement | Medium | ~20ms |
| Precision | Slow motion, yoga | Very Low | ~60ms |

## License

MIT
