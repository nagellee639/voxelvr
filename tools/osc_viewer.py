#!/usr/bin/env python3
"""
VoxelVR OSC Viewer
==================

A simple standalone visualiser for VRChat OSC tracking data.
Shows a rotating 3D view of the trackers being sent.

Usage:
    python3 tools/osc_viewer.py --port 9000
"""

import argparse
import sys
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pythonosc import dispatcher, osc_server

# Mappings
TRACKER_ID_TO_NAME = {
    1: 'hip',
    2: 'chest',
    3: 'left_foot',
    4: 'right_foot',
    5: 'left_knee',
    6: 'right_knee',
    7: 'left_elbow',
    8: 'right_elbow'
}

# Simple connections for visualization
BONES = [
    ('chest', 'hip'),
    ('hip', 'left_knee'),
    ('left_knee', 'left_foot'),
    ('hip', 'right_knee'),
    ('right_knee', 'right_foot'),
    ('chest', 'left_elbow'),
    ('chest', 'right_elbow'),
]

class OSCVisualizer:
    def __init__(self, ip="0.0.0.0", port=9000):
        self.ip = ip
        self.port = port
        
        # Store tracker state: name -> {'pos': [x,y,z], 'rot': [x,y,z], 'last_update': time}
        self.trackers = {}
        self.lock = threading.Lock()
        
        # Setup OSC
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/tracking/trackers/*/position", self.handle_position)
        self.dispatcher.map("/tracking/trackers/*/rotation", self.handle_rotation)
        self.dispatcher.map("/tracking/trackers/head/position", self.handle_head_pos)
        self.dispatcher.map("/tracking/trackers/head/rotation", self.handle_head_rot)
        
        # Setup Server
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), self.dispatcher)
        print(f"Listening on {ip}:{port}...")
        
        # Setup Plot
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(elev=15, azim=45)
        
        # Animation state
        self.start_time = time.time()
    
    def handle_position(self, address, *args):
        try:
            # Address format: /tracking/trackers/ID/position
            parts = address.split('/')
            tracker_id = int(parts[3])
            name = TRACKER_ID_TO_NAME.get(tracker_id, f"unknown_{tracker_id}")
            
            with self.lock:
                if name not in self.trackers:
                    self.trackers[name] = {'pos': [0,0,0], 'rot': [0,0,0], 'last_update': 0}
                self.trackers[name]['pos'] = args
                self.trackers[name]['last_update'] = time.time()
        except Exception as e:
            print(f"Error parsing pos: {e}")

    def handle_rotation(self, address, *args):
        try:
            parts = address.split('/')
            tracker_id = int(parts[3])
            name = TRACKER_ID_TO_NAME.get(tracker_id, f"unknown_{tracker_id}")
            
            with self.lock:
                if name not in self.trackers:
                    self.trackers[name] = {'pos': [0,0,0], 'rot': [0,0,0], 'last_update': 0}
                self.trackers[name]['rot'] = args
                self.trackers[name]['last_update'] = time.time()
        except Exception:
            pass

    def handle_head_pos(self, address, *args):
        # Handle head specifically
        with self.lock:
            if 'head' not in self.trackers:
                self.trackers['head'] = {'pos': [0,0,0], 'rot': [0,0,0], 'last_update': 0}
            self.trackers['head']['pos'] = args
            self.trackers['head']['last_update'] = time.time()

    def handle_head_rot(self, address, *args):
        with self.lock:
            if 'head' not in self.trackers:
                self.trackers['head'] = {'pos': [0,0,0], 'rot': [0,0,0], 'last_update': 0}
            self.trackers['head']['rot'] = args
            self.trackers['head']['last_update'] = time.time()

    def start(self):
        # Start OSC server in thread
        server_thread = threading.Thread(target=self.server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # Start Animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=50)
        plt.show()

    def update_plot(self, frame):
        self.ax.clear()
        
        # Rotate view
        angle = (time.time() - self.start_time) * 30  # 30 deg/sec
        self.ax.view_init(elev=15, azim=angle)
        
        # Set limits (approx VRChat scale: 2m tall, 1m wide)
        self.ax.set_xlim(-0.8, 0.8)
        self.ax.set_ylim(-0.8, 0.8) # Depth
        self.ax.set_zlim(-0.1, 2.0) # Up
        
        # Unity Coordinates: X Right, Y Up, Z Forward
        # Matplotlib: Defaults usually Z up.
        # We need to map:
        # Unity X -> Mpl X
        # Unity Y -> Mpl Z (Up)
        # Unity Z -> Mpl Y (Depth)
        
        with self.lock:
            # Filter stale trackers (> 1 sec old)
            now = time.time()
            active_trackers = {
                k: v for k,v in self.trackers.items() 
                if now - v['last_update'] < 1.0
            }
            
            # Scatter points
            xs, ys, zs = [], [], []
            colors = []
            
            if not active_trackers:
                self.ax.text(0, 1.0, 0, "No data...", ha='center')
                return
            
            coords = {} # name -> (mx, my, mz)
            
            for name, data in active_trackers.items():
                ux, uy, uz = data['pos']
                
                # Conversion Unity -> Matplotlib (Z-up)
                # Unity: X-Right, Y-Up, Z-Forward
                # Matplotlib Default: X/Y ground, Z up
                
                mx = ux
                my = uz 
                mz = uy
                
                coords[name] = (mx, my, mz)
                
                xs.append(mx)
                ys.append(my)
                zs.append(mz)
                
                # Color code
                if name == 'head': colors.append('red')
                elif 'left' in name: colors.append('cyan')
                elif 'right' in name: colors.append('magenta')
                else: colors.append('yellow')
                
                # Label
                self.ax.text(mx, my, mz, name, fontsize=8)
            
            self.ax.scatter(xs, ys, zs, c=colors, s=50)
            
            # Draw bones
            for start, end in BONES:
                if start in coords and end in coords:
                    p1 = coords[start]
                    p2 = coords[end]
                    self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='white', alpha=0.5)

            # Head connection
            if 'head' in coords and 'chest' in coords:
                p1 = coords['head']
                p2 = coords['chest']
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='white', alpha=0.5)
                
        # Axis labels
        self.ax.set_xlabel('X (Right)')
        self.ax.set_ylabel('Z (Fwd)')
        self.ax.set_zlabel('Y (Up)')

def main():
    parser = argparse.ArgumentParser(description="VoxelVR OSC Viewer")
    parser.add_argument("--port", type=int, default=9000, help="OSC Listen Port")
    args = parser.parse_args()
    
    viz = OSCVisualizer(port=args.port)
    viz.start()

if __name__ == "__main__":
    main()
