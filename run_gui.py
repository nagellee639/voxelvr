#!/usr/bin/env python3
import os
os.environ["MPLBACKEND"] = "Agg"
"""
VoxelVR GUI Application

Main entry point for the graphical user interface.

Usage:
    python run_gui.py [options]

This provides a complete interface for:
- Camera preview and positioning
- ChArUco board calibration
- Full body tracking
- Performance monitoring
- Debug and parameter tuning
"""

import argparse
import sys
import time
import threading
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from voxelvr.config import VoxelVRConfig, CameraConfig, MultiCameraCalibration
from voxelvr.gui import VoxelVRApp
from voxelvr.gui.unified_app import UnifiedVoxelVRApp
from voxelvr.gui.performance_panel import PerformanceMetrics


def run_tracking_thread(app: VoxelVRApp, config: VoxelVRConfig, calibration: MultiCameraCalibration):
    """
    Background thread for running the tracking pipeline.
    
    This is started when the user clicks "Start Tracking" in the GUI.
    """
    from voxelvr.capture import CameraManager
    from voxelvr.pose import PoseDetector2D, PoseFilter
    from voxelvr.pose.triangulation import TriangulationPipeline, compute_projection_matrices
    from voxelvr.transport import OSCSender
    from voxelvr.transport.osc_sender import pose_to_trackers_with_rotations
    from voxelvr.transport.coordinate import create_default_transform, transform_pose_to_vrchat
    from voxelvr.gui.skeleton_viewer import get_tpose
    from voxelvr.config import CameraIntrinsics, CameraExtrinsics
    import numpy as np
    
    # Give a small delay to ensure preview cameras are fully released
    time.sleep(0.5)
    
    # Get configuration from GUI
    osc_ip, osc_port = app.tracking_panel.get_osc_config()
    enabled_trackers = app.tracking_panel.get_enabled_trackers()
    
    print(f"Debug: Initial enabled trackers: {enabled_trackers}")
    if not enabled_trackers:
        print("Warning: No trackers enabled! Forcing all enabled for debugging.")
        enabled_trackers = ["hip", "chest", "left_foot", "right_foot", "left_knee", "right_knee", "left_elbow", "right_elbow"]
    
    # Setup camera manager
    camera_ids = list(calibration.cameras.keys())
    camera_configs = [
        CameraConfig(id=cam_id, resolution=(1280, 720), fps=30)
        for cam_id in camera_ids
    ]
    
    camera_manager = None
    detector = None
    osc_sender = None
    
    try:
        print("Initializing camera manager...")
        
        # Check if app already has an active camera manager (UnifiedVoxelVRApp)
        if hasattr(app, 'camera_manager') and app.camera_manager:
            print("Using existing camera manager from app")
            camera_manager = app.camera_manager
            # Ensure it has the right calibration
            camera_manager.load_calibration(calibration)
        else:
            # Create new manager (legacy/cli mode)
            camera_manager = CameraManager(camera_configs)
            camera_manager.load_calibration(calibration)
        
        
        # Load pose detector
        print("Loading pose detector model...")
        detector = PoseDetector2D(confidence_threshold=0.3)
        if not detector.load_model():
            app.tracking_panel.on_tracking_error("Failed to load pose model")
            return
        
        # Build projection matrices
        intrinsics_list = []
        extrinsics_list = []
        
        for cam_id, data in calibration.cameras.items():
            if cam_id in camera_ids:
                intrinsics_list.append(CameraIntrinsics(**data['intrinsics']))
                extrinsics_list.append(CameraExtrinsics(**data['extrinsics']))
        
        projection_matrices = compute_projection_matrices(intrinsics_list, extrinsics_list)
        
        # Create pipeline components
        triangulation = TriangulationPipeline(projection_matrices, confidence_threshold=0.3)
        
        # Get filter parameters from debug panel
        optimizer = app.debug_panel.optimizer
        pose_filter = PoseFilter(
            num_joints=17,
            min_cutoff=optimizer.min_cutoff,
            beta=optimizer.beta,
            d_cutoff=optimizer.d_cutoff,
        )
        
        # Connect parameter changes to filter
        def on_params_change(min_cutoff, beta, d_cutoff):
            pose_filter.update_parameters(min_cutoff, beta, d_cutoff)
        
        app.debug_panel.add_param_callback(on_params_change)
        
        # Coordinate transform
        coord_transform = create_default_transform()
        
        # OSC sender
        print("Connecting OSC...")
        if hasattr(app, 'osc_sender') and app.osc_sender:
             print("Using existing OSC sender from app")
             osc_sender = app.osc_sender
        else:
             osc_sender = OSCSender(ip=osc_ip, port=osc_port, send_rate=60.0)
             if not osc_sender.connect():
                 app.tracking_panel.on_tracking_error("Failed to connect OSC")
                 return
        
        # Configure enabled trackers
        for tracker_name in ['hip', 'chest', 'left_foot', 'right_foot', 
                              'left_knee', 'right_knee', 'left_elbow', 'right_elbow']:
            osc_sender.enable_tracker(tracker_name, tracker_name in enabled_trackers)
        
        # Start cameras
        print("Starting cameras...")
        if not camera_manager.start_all():
            app.tracking_panel.on_tracking_error("Failed to start cameras")
            return
        
        # Notify started - this updates the GUI
        print("Tracking started!")
        app.tracking_panel.on_tracking_started()
        app.osc_status.on_connect()
        
        # Main tracking loop
        frame_count = 0
        start_time = time.time()
        
        while app.tracking_panel.is_running and app.state.is_running:
            loop_start = time.time()
            
            # Capture frames
            frames_raw = camera_manager.get_synchronized_frames()
            if not frames_raw:
                continue
            
            frames = {cam_id: f.image for cam_id, f in frames_raw.items()}
            capture_time = time.time() - loop_start
            
            # 2D pose detection and GUI Update
            detect_start = time.time()
            keypoints_2d = {}
            
            for cam_id, frame in frames.items():
                kp = detector.detect(frame, camera_id=cam_id)
                if kp:
                    keypoints_2d[cam_id] = kp
            
            # Send 2D detections to GUI for visualization (avoids double inference)
            if hasattr(app, 'update_2d_detections'):
                app.update_2d_detections(keypoints_2d)
            
            detect_time = time.time() - detect_start
            
            # 3D triangulation
            triang_start = time.time()
            pose_3d = None
            
            # Debug: Print detection status every 60 frames
            if frame_count % 60 == 0:
                det_status = [f"Cam{k}:{len(v.positions) if hasattr(v, 'positions') else 0}" for k,v in keypoints_2d.items()]
                print(f"Debug [{frame_count}]: Detections in {len(keypoints_2d)} cams: {det_status}")
            
            if len(keypoints_2d) >= 2:
                kp_list = list(keypoints_2d.values())
                try:
                    pose_3d = triangulation.process(kp_list)
                    
                    # Debug: Print triangulation result
                    if frame_count % 60 == 0:
                        if pose_3d:
                            valid_cnt = np.sum(pose_3d['valid'])
                            print(f"Debug [{frame_count}]: Triangulation produced {valid_cnt} valid joints")
                        else:
                             print(f"Debug [{frame_count}]: Triangulation returned None")
                except Exception as e:
                    print(f"Triangulation Error: {e}")
            else:
                 if frame_count % 60 == 0:
                    print(f"Debug [{frame_count}]: Skipping triangulation - need 2+ cameras, have {len(keypoints_2d)}")
            
            # Fallback for debug: T-Pose if no data
            if pose_3d is None:
                pose_3d = get_tpose()
                if frame_count % 60 == 0:
                    print(f"Debug [{frame_count}]: Using fallback T-Pose")

            triang_time = time.time() - triang_start
            
            # Apply temporal filter
            filter_start = time.time()
            if pose_3d:
                pose_3d['positions'] = pose_filter.filter(
                    pose_3d['positions'],
                    pose_3d['valid'],
                )
                
                # Update debug panel
                app.debug_panel.update(
                    pose_3d['positions'],
                    pose_3d['valid'],
                )
            filter_time = time.time() - filter_start
            
            # Transform and send via OSC
            osc_start = time.time()
            trackers_sent = 0
            
            if pose_3d and np.any(pose_3d['valid']):
                # Transform coordinates
                transformed, _ = transform_pose_to_vrchat(
                    pose_3d['positions'],
                    coord_transform,
                )
                
                # Convert to VRChat trackers
                trackers = pose_to_trackers_with_rotations(
                    transformed,
                    pose_3d['confidences'],
                    pose_3d['valid'],
                )
                
                # Filter to enabled trackers
                trackers = {k: v for k, v in trackers.items() if k in enabled_trackers}
                
                # Debug OSC every 60 frames
                if frame_count % 60 == 0:
                     print(f"Debug [{frame_count}]: Sending {len(trackers)} trackers: {list(trackers.keys())}")
                
                # Send via OSC
                if osc_sender.send_all_trackers(trackers):
                    app.osc_status.on_message_sent()
                    trackers_sent = len(trackers)
                
                # Update tracking panel
                app.tracking_panel.update_pose(
                    pose_3d['positions'],
                    pose_3d['valid'],
                    pose_3d['confidences'],
                )
            
            osc_time = time.time() - osc_start
            total_time = time.time() - loop_start
            
            # Update performance metrics
            metrics = PerformanceMetrics(
                capture_fps=1.0 / capture_time if capture_time > 0 else 0,
                detection_fps=1.0 / detect_time if detect_time > 0 else 0,
                triangulation_fps=1.0 / triang_time if triang_time > 0 else 0,
                filter_fps=1.0 / filter_time if filter_time > 0 else 0,
                total_fps=1.0 / total_time if total_time > 0 else 0,
                capture_latency_ms=capture_time * 1000,
                detection_latency_ms=detect_time * 1000,
                triangulation_latency_ms=triang_time * 1000,
                filter_latency_ms=filter_time * 1000,
                osc_latency_ms=osc_time * 1000,
                total_latency_ms=total_time * 1000,
                num_valid_joints=np.sum(pose_3d['valid']) if pose_3d else 0,
                avg_confidence=np.mean(pose_3d['confidences'][pose_3d['valid']]) if pose_3d and np.any(pose_3d['valid']) else 0,
                jitter_mm=app.debug_panel.metrics.jitter_position_mm,
            )
            
            app.update_performance(metrics)
            
            # Update tracking status
            app.tracking_panel.update_status(
                fps=metrics.total_fps,
                valid_joints=metrics.num_valid_joints,
                trackers_sending=trackers_sent,
            )
            
            frame_count += 1
        
    except Exception as e:
        app.tracking_panel.on_tracking_error(str(e))
    finally:
        # Cleanup
        if 'camera_manager' in locals() and camera_manager:
            # Only stop if it's NOT the shared app manager
            is_shared = hasattr(app, 'camera_manager') and app.camera_manager is camera_manager
            if not is_shared:
                camera_manager.stop_all()
            else:
                print("Leaving shared camera manager running")
        if 'osc_sender' in locals() and osc_sender:
             # Only disconnect if we created it privately
            if not (hasattr(app, 'osc_sender') and app.osc_sender is osc_sender):
                osc_sender.disconnect()
        
        app.tracking_panel.on_tracking_stopped()
        app.osc_status.on_disconnect()


def main():
    parser = argparse.ArgumentParser(description="VoxelVR GUI Application")
    parser.add_argument(
        "--calibration", "-C",
        type=Path,
        default=None,
        help="Path to calibration.json file"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Window height"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy multi-panel interface instead of unified view"
    )
    parser.add_argument(
        "--record", "-r",
        action="store_true",
        help="Record time-synced video from all cameras"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = VoxelVRConfig.load()
    config.ensure_dirs()
    
    # Load calibration if exists
    calibration = None
    calibration_path = args.calibration or (config.calibration_dir / "calibration.json")
    
    if calibration_path.exists():
        print(f"Loading calibration: {calibration_path}")
        calibration = MultiCameraCalibration.load(calibration_path)
    
    # Create application (unified is now the default)
    if args.legacy:
        print("Starting legacy interface...")
        app = VoxelVRApp(
            title="VoxelVR - Full Body Tracking",
            width=args.width,
            height=args.height,
        )
    else:
        print("Starting unified interface...")
        app = UnifiedVoxelVRApp(
            title="VoxelVR - Unified Tracking",
            width=args.width,
            height=args.height,
        )
        
        # Enable recording if requested
        if args.record:
            app.enable_recording()
    
    # Setup tracking callbacks
    tracking_thread = None
    
    def on_start_tracking():
        nonlocal tracking_thread, calibration
        
        # For unified app, get calibration from the calibration panel
        active_calibration = calibration
        if hasattr(app, 'calibration_panel') and app.calibration_panel.state.extrinsics.result:
            active_calibration = app.calibration_panel.state.extrinsics.result
            print(f"Using computed calibration with {len(active_calibration.cameras)} cameras")
        
        if active_calibration is None:
            print("Tracking error: No calibration loaded")
            # For unified app, don't use tracking_panel (it doesn't exist)
            if hasattr(app, 'tracking_panel'):
                app.tracking_panel.on_tracking_error("No calibration loaded")
            return
        
        tracking_thread = threading.Thread(
            target=run_tracking_thread,
            args=(app, config, active_calibration),
            daemon=True,
        )
        tracking_thread.start()
    
    def on_stop_tracking():
        # Thread will exit on its own when is_running becomes False
        pass
    
    app.set_tracking_callbacks(
        on_start=on_start_tracking,
        on_stop=on_stop_tracking,
    )
    
    # Run application
    print("Starting VoxelVR GUI...")
    print("Press Ctrl+C or close window to exit.")
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        app.request_stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
