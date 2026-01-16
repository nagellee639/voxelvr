#!/usr/bin/env python3
"""
Test Data Downloader

Downloads real-world images for testing detection pipelines.
"""

import urllib.request
import urllib.error
from pathlib import Path
import sys

def download_file(url: str, dest: Path):
    """Download a file with user-agent header to avoid blocking."""
    print(f"Downloading {url}...")
    try:
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        with urllib.request.urlopen(req) as response, open(dest, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"  Saved to {dest} ({len(data)} bytes)")
        return True
    except Exception as e:
        print(f"  Failed to download: {e}")
        return False

def main():
    output_dir = Path("test_data/external")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Person Image (for pose tracking)
    # Source: OpenCV Samples (Messi)
    person_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg"
    download_file(person_url, output_dir / "person.jpg")
    
    # 2. ChArUco Board (for calibration)
    # Source: GitHub repo with a board image
    # Note: If this fails, we will fallback to generated image in tests, but we try real first.
    charuco_url = "https://raw.githubusercontent.com/ductmanhn/ChArUco-Board-Calibration/master/board.png"
    download_file(charuco_url, output_dir / "charuco_external.png")

if __name__ == "__main__":
    main()
