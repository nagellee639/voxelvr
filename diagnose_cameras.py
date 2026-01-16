
import cv2
import glob
import sys
import time

def diagnose():
    print("Diagnosis: Starting camera check...")
    
    # 1. Check devices
    if sys.platform.startswith('linux'):
        devices = sorted(glob.glob('/dev/video*'))
        print(f"Found device files: {devices}")
        candidates = []
        for dev in devices:
            try:
                idx = int(dev.replace('/dev/video', ''))
                candidates.append(idx)
            except ValueError:
                pass
    else:
        print("Not on Linux, checking indices 0-5")
        candidates = list(range(6))

    print(f"Checking candidates: {candidates}")

    # 2. Probe each candidate
    for idx in candidates:
        print(f"\n--- Probimg Camera Index {idx} ---")
        cap = cv2.VideoCapture(idx)
        
        if not cap.isOpened():
            print(f"  [FAIL] Failed to open index {idx}")
            continue
            
        print(f"  [OK] Opened index {idx}")
        
        # Check backend
        backend = cap.getBackendName()
        print(f"  Backend: {backend}")
        
        # Read frame immediately
        ret, frame = cap.read()
        print(f"  Immediate read: {'Success' if ret else 'Failed'}")
        
        if not ret:
            print("  [INFO] Retrying with 1.0s warmup delay...")
            time.sleep(1.0)
            ret, frame = cap.read()
            print(f"  Delayed read: {'Success' if ret else 'Failed'}")
            
            if not ret:
                # Try reading multiple times (buffer flush)
                for i in range(5):
                    ret, frame = cap.read()
                    if ret:
                        print(f"  [OK] Succeeded on attempt {i+2}")
                        break
        
        if ret:
            h, w = frame.shape[:2]
            print(f"  Frame size: {w}x{h}")
        else:
            print("  [FAIL] Could not read any frame.")
            
        cap.release()

if __name__ == "__main__":
    diagnose()
