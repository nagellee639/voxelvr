import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, "/home/lee/voxelvr")

from voxelvr.pose import PoseDetector2D, ConfidenceFilter
from voxelvr.pose.detector_2d import Keypoints2D

def main():
    root = Path("/home/lee/voxelvr/dataset/tracking/20260116_165920")
    cam_folders = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("cam_")])
    
    print(f"Found camera folders: {[d.name for d in cam_folders]}")
    cam_ids = [int(d.name.split('_')[1]) for d in cam_folders]
    
    detector = PoseDetector2D(confidence_threshold=0.3)
    detector.load_model()
    
    conf_filter = ConfidenceFilter(
        confidence_threshold=0.3,
        grace_period_frames=7,
        reactivation_frames=3
    )
    
    # Process first 100 frames
    for i in range(100):
        keypoints_list = []
        filename = f"track_{i:04d}.jpg"
        
        detections_info = []
        
        for cam_idx, folder in enumerate(cam_folders):
            cam_id = cam_ids[cam_idx]
            path = folder / filename
            if not path.exists():
                break
                
            img = cv2.imread(str(path))
            if img is None:
                continue
                
            kp = detector.detect(img, camera_id=cam_id)
            if kp:
                keypoints_list.append(kp)
                detections_info.append(f"Cam{cam_id}:AvgConf={np.mean(kp.confidences):.2f}")

        if not keypoints_list:
            if i > 0 and len(cam_folders) > 0:
                print(f"End of stream at frame {i}")
            break
            
        filtered_kps, diagnostics = conf_filter.update(keypoints_list)
        
        status = "FAIL" if len(filtered_kps) < 2 else "PASS"
        print(f"Frame {i:03d}: Input={len(keypoints_list)} ({', '.join(detections_info)}) -> Output={len(filtered_kps)} [{status}]")
        
        # Debug internals if failing
        if len(filtered_kps) < 2 and i > 10:
             # Sample active views for joint 0 (nose)
             active_views = diagnostics['active_views_per_joint']
             print(f"    Active views per joint: {active_views[:5]}...")

if __name__ == "__main__":
    main()
