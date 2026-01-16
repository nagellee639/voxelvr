# VoxelVR Quick Start Guide

**Your Setup:**
- 🖥️ Ubuntu laptop with 3 USB webcams + 1 built-in webcam (4 total)
- 🌐 Tailscale VPN connecting laptop to desktop
- 🎮 Windows desktop running VRChat

---

## Step 1: Install VoxelVR on Your Laptop

```bash
cd /home/lee/voxelvr

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Choose Your Workflow

### Option A: GUI Mode (Recommended for First-Time Setup)

```bash
python3 run_gui.py
```

The GUI provides:
- **Camera Preview tab** - View all camera feeds for positioning
- **Calibration tab** - Step-by-step calibration wizard with ChArUco board export
- **Tracking tab** - Start/stop tracking with OSC config and tracker toggles
- **Debug tab** - Real-time filter tuning with auto-adjustment profiles

### Option B: CLI Mode (For Automation/Scripting)

Continue with the CLI steps below (Step 3 onwards).

---

## Step 3: Configure VRChat on Windows Desktop

### 2.1 Enable OSC in VRChat

1. **Launch VRChat** on your Windows desktop
2. Open **Action Menu** (press R on desktop or use menu button in VR)
3. Navigate to: **Options → OSC → Enable**
4. Verify **OSC** toggle is **ON**

![OSC Location](https://docs.vrchat.com/docs/osc-overview)

### 2.2 Configure Full Body Tracking in VRChat

1. In VRChat Action Menu: **Options → Tracking & IK**
2. Set **Tracking Type** to: **6-Point Tracking** or **Generic Trackers**
3. Enable **OSC Trackers**

### 2.3 Allow OSC Through Windows Firewall

**Option A: Using PowerShell (Run as Administrator)**
```powershell
netsh advfirewall firewall add rule name="VRChat OSC" dir=in action=allow protocol=udp localport=9000
```

**Option B: Using Windows UI**
1. Search for "Windows Defender Firewall" in Start menu
2. Click "Allow an app through firewall"
3. Click "Change settings" → "Allow another app"
4. Browse to VRChat.exe or add UDP port 9000

---

## Step 3: Get Your Tailscale IP

**On your Windows desktop:**
1. Open PowerShell or Command Prompt
2. Run:
   ```powershell
   tailscale ip -4
   ```
3. Note the IP (looks like `100.x.x.x`)

**Or use the Tailscale app:**
- Click the Tailscale icon in system tray
- Your IP is shown in the menu

---

## Step 4: Test the Connection

**On your Ubuntu laptop:**
```bash
# Test that Tailscale connection works
ping YOUR_TAILSCALE_IP

# Test OSC connection specifically
cd /home/lee/voxelvr
source venv/bin/activate
python3 -m voxelvr.transport.discovery --test --ip YOUR_TAILSCALE_IP
```

You should see:
```
✓ Sent test messages to 100.x.x.x:9000
  Check VRChat - if OSC is enabled, you should see tracker activity
```

---

## Step 5: Detect Your Cameras

```bash
# List available cameras
python3 -c "
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Camera {i}: {w}x{h}')
        cap.release()
"
```

Example output:
```
Camera 0: 1280x720    # Built-in webcam
Camera 1: 1920x1080   # USB webcam 1
Camera 2: 1920x1080   # USB webcam 2
Camera 3: 1280x720    # USB webcam 3
```

---

## Step 6: Position Your Cameras

For best tracking, place cameras:

```
        [Cam 2 - Back]
             ▲
             │
    [Cam 1]  │  [Cam 3]
       ◄─────┼─────►
             │
             │
        [You Stand Here]
             │
             ▼
        [Cam 0 - Front]
        (built-in webcam on laptop/tripod)
```

**Tips:**
- Place cameras at chest height (~1.2-1.5m)
- Point all cameras toward center of tracking area
- Keep at least 2m between cameras and tracking center
- Ensure good lighting (avoid backlight)

---

## Step 7: Calibrate Your Cameras

### 7.1 Print the Calibration Board

```bash
# Generate ChArUco board PDF
python3 run_calibration.py --generate-board

# Print charuco_board.pdf at 100% scale (no scaling)
```

### 7.2 Run Intrinsic Calibration (One Camera at a Time)

```bash
# Calibrate each camera
python3 run_calibration.py --intrinsics --camera 0
python3 run_calibration.py --intrinsics --camera 1
python3 run_calibration.py --intrinsics --camera 2
python3 run_calibration.py --intrinsics --camera 3
```

For each camera:
1. Hold the printed board in front of the camera
2. Move it around to different positions/angles
3. Press SPACE to capture each position
4. Capture ~20 frames from different angles
5. Press Q when done

### 7.3 Run Extrinsic Calibration (All Cameras Together)

```bash
python3 run_calibration.py --extrinsics --cameras 0,1,2,3
```

1. Place the board on the floor in the center of your tracking area
2. Ensure ALL cameras can see the board
3. Press SPACE to capture
4. Move the board and capture from 3-5 positions
5. Press Q when done

---

## Step 8: Run Tracking!

```bash
# Start tracking and send to VRChat
python3 run_tracking.py \
    --cameras 0,1,2,3 \
    --osc-ip YOUR_TAILSCALE_IP \
    --osc-port 9000 \
    --preview

# Example with your Tailscale IP:
python3 run_tracking.py --cameras 0,1,2,3 --osc-ip 100.64.123.45 --preview
```

---

## Step 9: Calibrate in VRChat

1. In VRChat, enter a world with an avatar
2. Stand in a T-pose in your tracking area
3. In Action Menu: **Options → Tracking & IK → Calibrate**
4. Hold the pose while VRChat calibrates

Your avatar should now follow your movements!

---

## Troubleshooting

### "No trackers appearing in VRChat"

1. **Check OSC is enabled** in VRChat settings
2. **Check firewall** on Windows:
   ```powershell
   # Test if port is blocked
   Test-NetConnection -ComputerName localhost -Port 9000
   ```
3. **Verify Tailscale connection:**
   ```bash
   # From laptop
   ping YOUR_DESKTOP_TAILSCALE_IP
   ```

### "Tracking is jittery"

1. **Improve lighting** - more light = better detection
2. **Adjust filter settings:**
   ```bash
   python3 run_tracking.py --filter-beta 0.3 ...  # Lower = smoother
   ```
3. **Check camera FPS** - ensure cameras are at 30fps

### "Some body parts don't track"

1. **Wear contrasting clothing** - solid colors, not patterns
2. **Ensure all cameras can see you**
3. **Check camera coverage** - you need 2+ cameras seeing each joint

### "High latency"

Your 50ms Tailscale ping is acceptable. Check:
1. **Camera processing** - lower resolution helps:
   ```bash
   python3 run_tracking.py --resolution 640x480 ...
   ```
2. **CPU usage** - close other applications

---

## Commands Reference

```bash
# Test without cameras (synthetic data)
python3 run_demo.py --synthetic

# Preview camera feeds
python3 run_demo.py --cameras 0,1,2,3

# Run with visualization
python3 run_tracking.py --cameras 0,1,2,3 --osc-ip IP --preview

# Run headless (no display)
python3 run_tracking.py --cameras 0,1,2,3 --osc-ip IP

# Adjust smoothing
python3 run_tracking.py ... --filter-min-cutoff 0.5 --filter-beta 0.3

# Check system
python3 run_tests.py --platform-check
```

---

## VRChat OSC Address Reference

VoxelVR sends to these OSC addresses:

| Tracker | Address | Data |
|---------|---------|------|
| Hip | `/tracking/trackers/1/position` | [x, y, z] |
| Chest | `/tracking/trackers/2/position` | [x, y, z] |
| Left Foot | `/tracking/trackers/3/position` | [x, y, z] |
| Right Foot | `/tracking/trackers/4/position` | [x, y, z] |
| Left Knee | `/tracking/trackers/5/position` | [x, y, z] |
| Right Knee | `/tracking/trackers/6/position` | [x, y, z] |
| Left Elbow | `/tracking/trackers/7/position` | [x, y, z] |
| Right Elbow | `/tracking/trackers/8/position` | [x, y, z] |

Plus rotation: `/tracking/trackers/N/rotation` → [x, y, z] Euler degrees

---

## Need Help?

1. Run sanity checks:
   ```bash
   python3 -m pytest tests/test_sanity.py -v -s
   ```

2. Check system compatibility:
   ```bash
   python3 run_tests.py --platform-check
   ```

3. Test with synthetic data first:
   ```bash
   python3 run_demo.py --synthetic --osc-ip YOUR_IP
   ```
