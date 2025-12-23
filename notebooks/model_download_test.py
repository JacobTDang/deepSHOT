"""
Test script to understand how ML models download and cache

This will show you the difference between downloading libraries vs models
"""
import os
from pathlib import Path

print("=== Testing Model Downloads ===\n")

# 1. YOLOv8 Download Test
print("1. YOLOv8 Model Download:")
try:
    from ultralytics import YOLO
    
    # This will download the model automatically on first run
    # Look at the output - it shows download progress and cache location
    print("Loading YOLOv8s model...")
    model = YOLO("yolov8s.pt")  # ~22MB download
    
    # Show where it's cached
    print(f"Model loaded successfully!")
    print(f"Model is cached at: ~/.ultralytics/")
    
    # Show model info
    print(f"Model classes: {len(model.names)} object types")
    print(f"Some examples: {list(model.names.values())[:5]}")
    
except Exception as e:
    print(f"Error loading YOLO: {e}")

print("\n" + "="*50 + "\n")

# 2. MediaPipe Download Test  
print("2. MediaPipe Model Download:")
try:
    import mediapipe as mp
    
    # MediaPipe downloads models automatically too
    print("Initializing MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=lite, 1=full, 2=heavy
        min_detection_confidence=0.5
    )
    
    print("MediaPipe Pose initialized successfully!")
    print("Model downloaded and cached internally by MediaPipe")
    
except Exception as e:
    print(f"Error loading MediaPipe: {e}")

print("\n" + "="*50 + "\n")

# 3. Show cache locations
print("3. Where Models Get Cached:")
print("YOLOv8: ~/.ultralytics/ (user home directory)")
print("MediaPipe: Internal to package installation")
print("PyTorch models: ~/.cache/torch/hub/")

print("\n=== Key Learning Points ===")
print("1. First run = downloads models (slower)")
print("2. Subsequent runs = uses cached models (faster)")
print("3. Models are separate from code - can be swapped/updated")
print("4. Different frameworks have different caching strategies")