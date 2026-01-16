# Network Setup Guide

This guide explains how to run VoxelVR tracking on one computer (e.g., laptop with cameras) and send the tracking data to VRChat running on another computer (desktop or Quest).

## Architecture

```
┌─────────────────┐     WiFi/Ethernet      ┌─────────────────┐
│  LAPTOP         │ ──────────────────────▶│  DESKTOP/QUEST  │
│  - Cameras      │        OSC UDP         │  - VRChat       │
│  - VoxelVR      │     Port 9000          │  - Avatar       │
└─────────────────┘                        └─────────────────┘
```

## Quick Start

### 1. Find Your VRChat Computer's IP Address

**On Windows (Desktop):**
```cmd
ipconfig
```
Look for "IPv4 Address" under your active network adapter (e.g., `192.168.1.100`)

**On Quest (Standalone):**
- Settings → Wi-Fi → Click your network → View IP address
- Or use the Quest browser to visit `whatismyip.com`

**On Quest via Link:**
- When connected via Link, use the desktop's IP address

### 2. Enable OSC in VRChat

1. Open VRChat
2. Action Menu → Options → OSC → Enable OSC
3. The default port is 9000

### 3. Run VoxelVR with Network Target

```bash
# On your laptop (with cameras)
python run_tracking.py --osc-ip 192.168.1.100 --osc-port 9000
```

Replace `192.168.1.100` with your VRChat computer's IP.

## Network Configuration

### Firewall Settings

**Windows (on the VRChat computer):**
```powershell
# Allow incoming UDP on port 9000
netsh advfirewall firewall add rule name="VRChat OSC" dir=in action=allow protocol=udp localport=9000
```

**Linux:**
```bash
sudo ufw allow 9000/udp
```

### Quest Standalone Notes

> ⚠️ **Important**: Quest standalone VRChat may not accept OSC from external devices by default.

For Quest standalone:
1. Use Quest Link or Air Link to stream from a PC
2. Run VRChat on the PC, not standalone
3. Send OSC to the PC's IP address

### Same Network Requirement

Both computers must be on the **same local network** (same WiFi or connected via Ethernet to same router).

### Tailscale / VPN Setup

Tailscale works great for VoxelVR! It creates a virtual network between your devices, even across different physical networks (like campus WiFi → home desktop).

**Benefits of Tailscale:**
- Works across different networks (laptop on campus WiFi, desktop at home)
- Low overhead (~1-2ms added latency)
- Encrypted traffic
- No port forwarding needed

**Setup:**
1. Install Tailscale on both laptop and desktop: https://tailscale.com/download
2. Sign in on both devices with the same account
3. Find your desktop's Tailscale IP: `tailscale ip -4`
4. Use the Tailscale IP instead of local IP:

```bash
# On laptop, send to desktop via Tailscale
python run_tracking.py --osc-ip 100.x.x.x --osc-port 9000
```

**Latency considerations:**
- Your 50ms ping is acceptable for VRChat tracking
- Total latency budget: ~50ms (tracking) + 50ms (network) = ~100ms
- VRChat can handle up to ~150ms latency before noticeable lag

**Test your Tailscale connection:**
```bash
# From laptop, ping desktop via Tailscale
ping 100.x.x.x

# Test OSC specifically
python -m voxelvr.transport.discovery --test --ip 100.x.x.x
```

## Advanced Configuration

### Multiple Destinations

Send tracking data to multiple computers simultaneously:

```python
# In your custom script
from voxelvr.transport import OSCSender

# Send to both desktop and a recording PC
senders = [
    OSCSender(ip="192.168.1.100", port=9000),  # Desktop with VRChat
    OSCSender(ip="192.168.1.101", port=9000),  # Recording PC
]

for sender in senders:
    sender.connect()

# In your tracking loop
for sender in senders:
    sender.send_all_trackers(trackers)
```

### Network Discovery

Use the built-in network scanner to find VRChat instances:

```bash
python -m voxelvr.transport.discovery --scan
```

### Latency Optimization

For best results over WiFi:
- Use 5GHz WiFi (lower latency than 2.4GHz)
- Position laptop close to router
- Consider USB Ethernet adapter for wired connection
- Target < 10ms network latency (use `ping` to test)

### Testing Network Connection

Before running tracking, test that OSC messages reach VRChat:

```bash
python -m voxelvr.transport.test_connection --ip 192.168.1.100 --port 9000
```

This sends test messages and checks if VRChat responds.

## Troubleshooting

### "Connection refused" or messages not arriving

1. **Check firewall** on VRChat computer
2. **Verify IP address** is correct
3. **Confirm same network** (both on same WiFi)
4. **OSC enabled** in VRChat settings

### High latency / laggy tracking

1. **Check WiFi signal** strength
2. **Switch to 5GHz** WiFi if available
3. **Use wired Ethernet** for lowest latency
4. **Reduce camera resolution** to lower processing time

### Quest-specific issues

- Quest standalone has limited OSC support
- Use Quest Link to PC for best compatibility
- Ensure Quest and PC are on same network if using Air Link

## Example: Laptop → Desktop Setup

```bash
# Step 1: On desktop, find IP
ipconfig  # Note the IPv4 address, e.g., 192.168.1.50

# Step 2: On desktop, start VRChat with OSC enabled

# Step 3: On laptop, run tracking
cd /path/to/voxelvr
python run_tracking.py --osc-ip 192.168.1.50 --preview

# Step 4: In VRChat, calibrate full body tracking
# Your avatar should now move with your body!
```
