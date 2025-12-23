"""
Pose detection using MediaPipe
This is where we'll extract player body keypoints for technique analysis
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List


class PoseDetector:
    """
    Wraps MediaPipe pose detection for badminton analysis
    
    Think of this as an API that takes video frames and returns
    joint positions - similar to how your Android app might
    call a location API and get coordinates back.
    """
    
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        """
        Initialize pose detector
        
        Args:
            model_complexity: 0=lite, 1=full, 2=heavy (we'll start with 1)
            min_detection_confidence: How confident model needs to be (0.0-1.0)
        """
        # TODO: Initialize MediaPipe pose solution
        pass
    
    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect pose in a single frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Dictionary with pose landmarks or None if no person detected
        """
        # TODO: Implement pose detection
        pass
    
    def process_video(self, video_path: str) -> List[Dict]:
        """
        Process entire video and return pose data for each frame
        
        This is where the caching strategy becomes important -
        pose detection is expensive, so we save results to disk
        """
        # TODO: Implement video processing with caching
        pass