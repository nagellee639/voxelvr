
import os
import cv2
import time
import sys
import glob
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelvr.capture.manager import CameraManager
from voxelvr.capture.camera import Camera

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_frames(video_path, output_dir, interval=5, prefix="frame"):
    """Extract frames from video at given interval."""
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frame_name = f"{prefix}_{saved:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")
    return saved

def record_phase(manager, phase_name, output_dir, duration=None):
    """
    Record from all cameras for a specific phase.
    If duration is None, record until user stops (Ctrl+C).
    """
    ensure_dir(output_dir)
    print(f"\n--- Starting {phase_name} Phase ---")
    print(f"Saving to {output_dir}")
    
    # Initialize writers
    writers = {}
    fps = 30.0 # Assumption, will be updated from camera
    dims = (640, 480) # Assumption
    
    # Get actual cam params
    for cam_id, camera in manager.cameras.items():
        # Make sure camera is running
        if not camera.is_running:
            camera.start()
            time.sleep(1.0) # Warmup
            
        w, h = camera.resolution
        fps = camera.target_fps
        dims = (w, h)
        
        filename = os.path.join(output_dir, f"cam_{cam_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writers[cam_id] = cv2.VideoWriter(filename, fourcc, fps, dims)
        print(f"Camera {cam_id} recording to {filename} ({w}x{h} @ {fps}fps)")

    print("Recording... Press Ctrl+C to stop.")
    
    start_time = time.time()
    frame_counts = {cam_id: 0 for cam_id in manager.cameras}
    
    try:
        while True:
            # Check duration if set
            if duration and (time.time() - start_time > duration):
                print("Duration reached.")
                break
                
            # Get latest frames
            frames = manager.get_all_latest_frames()
            
            for cam_id, frame in frames.items():
                if cam_id in writers:
                    writers[cam_id].write(frame.image)
                    frame_counts[cam_id] += 1
            
            # Simple progress
            elapsed = time.time() - start_time
            sys.stdout.write(f"\rRecording: {elapsed:.1f}s | Frames: {sum(frame_counts.values())}")
            sys.stdout.flush()
            
            time.sleep(1.0 / (fps * 1.5)) # Sleep a bit less than frame time to poll faster

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        # Cleanup
        for writer in writers.values():
            writer.release()
            
    print(f"\n{phase_name} Phase Complete.")

def main():
    print("=== VoxelVR Dataset Recorder ===")
    
    # Setup directories
    session_id = get_session_id()
    base_dir = Path("dataset")
    raw_dir = base_dir / "raw" / session_id
    calib_dir = base_dir / "calibration" / session_id
    track_dir = base_dir / "tracking" / session_id
    
    ensure_dir(raw_dir)
    
    # detailed subdirs
    raw_calib_dir = raw_dir / "calibration"
    raw_move_dir = raw_dir / "movement"
    
    # 1. Setup Camera Manager
    print("Detecting cameras...")
    cam_ids = CameraManager.detect_cameras()
    if not cam_ids:
        print("No cameras found!")
        return

    print(f"Found cameras: {cam_ids}")
    
    # Create configs
    from voxelvr.config import CameraConfig
    configs = [CameraConfig(id=i, resolution=(1280, 720), fps=30) for i in cam_ids]
    
    manager = CameraManager(configs)
    
    try:
        manager.start_all()
        time.sleep(2.0) # Warmup
        
        # 2. Calibration Phase
        input("\nPress ENTER to start CALIBRATION recording (move board around)...")
        record_phase(manager, "Calibration", raw_calib_dir)
        
        # 3. Movement Phase
        input("\nPress ENTER to start MOVEMENT recording (move yourself around)...")
        record_phase(manager, "Movement", raw_move_dir)
        
    finally:
        manager.stop_all()
        print("\nCameras stopped.")

    # 4. Post-processing
    print("\n=== Post-Processing ===")
    
    # Process Calibration Videos
    print("Processing Calibration Videos...")
    if os.path.exists(raw_calib_dir):
        video_files = glob.glob(str(raw_calib_dir / "*.mp4"))
        for video_path in video_files:
            cam_id = Path(video_path).stem.replace("cam_", "")
            out_path = calib_dir / f"cam_{cam_id}"
            print(f"Extracting frames for Camera {cam_id}...")
            extract_frames(video_path, out_path, interval=10, prefix="calib")

    # Process Movement Videos
    print("Processing Movement Videos...")
    if os.path.exists(raw_move_dir):
        video_files = glob.glob(str(raw_move_dir / "*.mp4"))
        for video_path in video_files:
            cam_id = Path(video_path).stem.replace("cam_", "")
            out_path = track_dir / f"cam_{cam_id}"
            print(f"Extracting frames for Camera {cam_id}...")
            extract_frames(video_path, out_path, interval=5, prefix="track")

    print(f"\nDone! Dataset saved to {base_dir}")
    print(f"Session ID: {session_id}")

if __name__ == "__main__":
    main()
