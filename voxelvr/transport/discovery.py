"""
Network Discovery and Connection Testing

Utilities for finding VRChat instances on the network
and testing OSC connectivity.
"""

import socket
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class NetworkTarget:
    """A potential OSC target on the network."""
    ip: str
    port: int
    hostname: Optional[str] = None
    latency_ms: Optional[float] = None
    reachable: bool = False


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def get_network_prefix() -> str:
    """Get the network prefix (e.g., '192.168.1.')."""
    local_ip = get_local_ip()
    parts = local_ip.split('.')
    if len(parts) == 4:
        return '.'.join(parts[:3]) + '.'
    return "192.168.1."


def ping_host(ip: str, timeout: float = 0.5) -> Tuple[bool, float]:
    """
    Check if a host is reachable via ICMP-like check.
    
    Returns:
        Tuple of (reachable, latency_ms)
    """
    try:
        start = time.perf_counter()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Try to connect to a common port (this won't actually connect for OSC)
        # but tells us if the host is up
        result = sock.connect_ex((ip, 80))
        
        latency = (time.perf_counter() - start) * 1000
        sock.close()
        
        # Even if port 80 is closed, we got a response = host is up
        return True, latency
    except socket.timeout:
        return False, 0
    except Exception:
        return False, 0


def test_osc_port(ip: str, port: int = 9000, timeout: float = 1.0) -> bool:
    """
    Test if an OSC port appears to be open.
    
    Note: OSC uses UDP which is connectionless, so we can only
    verify the host is reachable, not that VRChat is listening.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        
        # Send a minimal OSC message
        # This is a valid OSC message for "/ping" with no arguments
        osc_ping = b'/ping\x00\x00\x00,\x00\x00\x00'
        
        sock.sendto(osc_ping, (ip, port))
        sock.close()
        return True
    except Exception:
        return False


def scan_network(
    port: int = 9000,
    timeout: float = 0.3,
    progress_callback=None,
) -> List[NetworkTarget]:
    """
    Scan local network for potential VRChat instances.
    
    Args:
        port: OSC port to check (default 9000)
        timeout: Timeout per host in seconds
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        List of reachable NetworkTarget objects
    """
    prefix = get_network_prefix()
    targets = []
    
    print(f"Scanning network {prefix}0/24 for OSC targets...")
    
    for i in range(1, 255):
        ip = f"{prefix}{i}"
        
        if progress_callback:
            progress_callback(i, 254)
        
        reachable, latency = ping_host(ip, timeout)
        
        if reachable:
            # Try to resolve hostname
            try:
                hostname = socket.gethostbyaddr(ip)[0]
            except:
                hostname = None
            
            # Test OSC port
            osc_ok = test_osc_port(ip, port, timeout)
            
            target = NetworkTarget(
                ip=ip,
                port=port,
                hostname=hostname,
                latency_ms=latency,
                reachable=osc_ok,
            )
            targets.append(target)
            
            status = "OSC OK" if osc_ok else "reachable"
            name = hostname or ip
            print(f"  Found: {name} ({ip}) - {status}, {latency:.1f}ms")
    
    return targets


def test_connection(ip: str, port: int = 9000) -> bool:
    """
    Test OSC connection to a specific target.
    
    Sends test messages and reports status.
    """
    from pythonosc import udp_client
    
    print(f"Testing OSC connection to {ip}:{port}...")
    
    try:
        client = udp_client.SimpleUDPClient(ip, port)
        
        # Send a few test messages
        for i in range(3):
            # Send position to tracker 1 (hip)
            client.send_message("/tracking/trackers/1/position", [0.0, 1.0, 0.0])
            client.send_message("/tracking/trackers/1/rotation", [0.0, 0.0, 0.0])
            time.sleep(0.1)
        
        print(f"✓ Sent test messages to {ip}:{port}")
        print("  Check VRChat - if OSC is enabled, you should see tracker activity")
        print("  (Avatar may twitch slightly)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send: {e}")
        return False


def interactive_setup() -> Optional[NetworkTarget]:
    """
    Interactive network setup wizard.
    
    Returns:
        Selected NetworkTarget or None if cancelled
    """
    print("\n" + "="*50)
    print("VoxelVR Network Setup")
    print("="*50)
    
    local_ip = get_local_ip()
    print(f"\nYour IP address: {local_ip}")
    
    print("\nOptions:")
    print("  1. Scan network for VRChat instances")
    print("  2. Enter IP address manually")
    print("  3. Use localhost (same computer)")
    print("  4. Cancel")
    
    choice = input("\nChoice [1-4]: ").strip()
    
    if choice == "1":
        targets = scan_network()
        
        if not targets:
            print("\nNo hosts found. Try entering IP manually.")
            return interactive_setup()
        
        print("\nFound targets:")
        for i, t in enumerate(targets):
            name = t.hostname or t.ip
            status = "✓" if t.reachable else "?"
            print(f"  {i+1}. {status} {name} ({t.ip}) - {t.latency_ms:.1f}ms")
        
        idx = input("\nSelect target [1-{}]: ".format(len(targets))).strip()
        try:
            return targets[int(idx) - 1]
        except:
            return None
    
    elif choice == "2":
        ip = input("Enter IP address: ").strip()
        port = input("Port [9000]: ").strip() or "9000"
        
        return NetworkTarget(ip=ip, port=int(port), reachable=True)
    
    elif choice == "3":
        return NetworkTarget(ip="127.0.0.1", port=9000, reachable=True)
    
    else:
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VoxelVR Network Tools")
    parser.add_argument('--scan', action='store_true', help='Scan network')
    parser.add_argument('--test', action='store_true', help='Test connection')
    parser.add_argument('--ip', type=str, help='Target IP address')
    parser.add_argument('--port', type=int, default=9000, help='OSC port')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Interactive setup')
    
    args = parser.parse_args()
    
    if args.interactive:
        target = interactive_setup()
        if target:
            print(f"\nSelected: {target.ip}:{target.port}")
            test_connection(target.ip, target.port)
    elif args.scan:
        targets = scan_network(args.port)
        print(f"\nFound {len(targets)} reachable hosts")
    elif args.test and args.ip:
        test_connection(args.ip, args.port)
    else:
        parser.print_help()
