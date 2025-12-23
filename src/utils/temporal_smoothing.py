"""
Temporal Smoothing Techniques for Pose Tracking

This module demonstrates different approaches to handle tracking instability.
Each method has different trade-offs between smoothness, latency, and accuracy.
"""
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import cv2


@dataclass
class PoseFrame:
    """Single frame of pose data"""
    timestamp: float
    landmarks: Optional[Dict[str, Tuple[float, float]]]  # {joint_name: (x, y)}
    confidence: float
    frame_id: int


class TemporalSmoother:
    """
    Base class for temporal smoothing algorithms
    
    Think of this like a filter that takes noisy input and produces smooth output.
    Similar to how you might smooth user input in a mobile app.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.pose_history = deque(maxlen=window_size)
    
    def add_frame(self, pose_frame: PoseFrame) -> Optional[PoseFrame]:
        """Add new frame and return smoothed result"""
        raise NotImplementedError
    
    def reset(self):
        """Clear history - call when starting new video/session"""
        self.pose_history.clear()


class LinearInterpolationSmoother(TemporalSmoother):
    """
    Method 1: Linear Interpolation
    
    When tracking is lost, draw a straight line between last known and next known positions.
    
    Pros: Simple, minimal latency
    Cons: Assumes linear motion (not realistic for sports)
    """
    
    def add_frame(self, pose_frame: PoseFrame) -> Optional[PoseFrame]:
        self.pose_history.append(pose_frame)
        
        if len(self.pose_history) < 2:
            return pose_frame  # Need at least 2 frames
        
        current = self.pose_history[-1]
        previous = self.pose_history[-2]
        
        # If current frame lost tracking but previous was good
        if current.landmarks is None and previous.landmarks is not None:
            # Look ahead to see if we can find next good frame
            # (In real-time, we'd need to buffer frames)
            return self._interpolate_missing_frame(previous, current)
        
        return current
    
    def _interpolate_missing_frame(self, prev_frame: PoseFrame, curr_frame: PoseFrame) -> PoseFrame:
        """Create interpolated frame between two known frames"""
        # For now, just return previous frame (simple fallback)
        # In full implementation, we'd interpolate between prev and next known frame
        interpolated = PoseFrame(
            timestamp=curr_frame.timestamp,
            landmarks=prev_frame.landmarks,  # Copy previous landmarks
            confidence=prev_frame.confidence * 0.5,  # Lower confidence for interpolated
            frame_id=curr_frame.frame_id
        )
        return interpolated


class KalmanFilterSmoother(TemporalSmoother):
    """
    Method 2: Kalman Filter
    
    Predicts where joints should be based on velocity and acceleration.
    Used in GPS navigation, missile guidance, and... pose tracking!
    
    Pros: Handles acceleration, very smooth
    Cons: More complex, can lag behind rapid changes
    """
    
    def __init__(self, window_size: int = 5):
        super().__init__(window_size)
        self.kalman_filters = {}  # One filter per joint
    
    def add_frame(self, pose_frame: PoseFrame) -> Optional[PoseFrame]:
        """Apply Kalman filtering to each joint independently"""
        if pose_frame.landmarks is None:
            return self._predict_missing_frame(pose_frame)
        
        # Update Kalman filters with new observations
        smoothed_landmarks = {}
        for joint_name, (x, y) in pose_frame.landmarks.items():
            if joint_name not in self.kalman_filters:
                self.kalman_filters[joint_name] = self._create_kalman_filter()
            
            # Predict and update
            kf = self.kalman_filters[joint_name]
            prediction = kf.predict()
            kf.correct(np.array([[x], [y]], dtype=np.float32))
            
            # Extract smoothed position
            smoothed_x = float(kf.statePost[0])
            smoothed_y = float(kf.statePost[1])
            smoothed_landmarks[joint_name] = (smoothed_x, smoothed_y)
        
        return PoseFrame(
            timestamp=pose_frame.timestamp,
            landmarks=smoothed_landmarks,
            confidence=pose_frame.confidence,
            frame_id=pose_frame.frame_id
        )
    
    def _create_kalman_filter(self):
        """Create Kalman filter for 2D position with velocity"""
        kf = cv2.KalmanFilter(4, 2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)
        
        # State transition matrix (position + velocity model)
        dt = 1.0 / 30.0  # Assume 30 FPS
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],   # x = x + vx*dt
            [0, 1, 0, dt],   # y = y + vy*dt
            [0, 0, 1, 0],    # vx = vx
            [0, 0, 0, 1]     # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Noise matrices (tuning parameters)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        return kf
    
    def _predict_missing_frame(self, pose_frame: PoseFrame) -> PoseFrame:
        """Use Kalman prediction when tracking is lost"""
        if not self.kalman_filters:
            return pose_frame  # No history to predict from
        
        predicted_landmarks = {}
        for joint_name, kf in self.kalman_filters.items():
            prediction = kf.predict()
            predicted_x = float(prediction[0])
            predicted_y = float(prediction[1])
            predicted_landmarks[joint_name] = (predicted_x, predicted_y)
        
        return PoseFrame(
            timestamp=pose_frame.timestamp,
            landmarks=predicted_landmarks,
            confidence=0.3,  # Low confidence for prediction
            frame_id=pose_frame.frame_id
        )


class MovingAverageSmoother(TemporalSmoother):
    """
    Method 3: Moving Average
    
    Average joint positions over last N frames.
    
    Pros: Very simple, reduces jitter
    Cons: Always lags behind real motion, can't handle missing frames well
    """
    
    def add_frame(self, pose_frame: PoseFrame) -> Optional[PoseFrame]:
        self.pose_history.append(pose_frame)
        
        # Only process frames with valid landmarks
        valid_frames = [f for f in self.pose_history if f.landmarks is not None]
        
        if len(valid_frames) < 2:
            return pose_frame
        
        # Average positions across valid frames
        averaged_landmarks = {}
        all_joints = set()
        for frame in valid_frames:
            all_joints.update(frame.landmarks.keys())
        
        for joint_name in all_joints:
            x_values = []
            y_values = []
            for frame in valid_frames:
                if joint_name in frame.landmarks:
                    x, y = frame.landmarks[joint_name]
                    x_values.append(x)
                    y_values.append(y)
            
            if x_values:  # If we have data for this joint
                avg_x = sum(x_values) / len(x_values)
                avg_y = sum(y_values) / len(y_values)
                averaged_landmarks[joint_name] = (avg_x, avg_y)
        
        return PoseFrame(
            timestamp=pose_frame.timestamp,
            landmarks=averaged_landmarks,
            confidence=pose_frame.confidence,
            frame_id=pose_frame.frame_id
        )


# Demo function to compare all three methods
def compare_smoothing_methods():
    """
    Educational function to understand trade-offs between different approaches
    """
    print("=== Temporal Smoothing Comparison ===\n")
    
    # Create test data with intentional gaps
    test_frames = [
        PoseFrame(0.0, {"wrist": (100, 100)}, 0.9, 0),
        PoseFrame(0.033, {"wrist": (102, 105)}, 0.9, 1),
        PoseFrame(0.066, None, 0.0, 2),  # Lost tracking!
        PoseFrame(0.099, None, 0.0, 3),  # Still lost!
        PoseFrame(0.132, {"wrist": (108, 115)}, 0.9, 4),
    ]
    
    # Test each smoother
    smoothers = {
        "Linear Interpolation": LinearInterpolationSmoother(),
        "Kalman Filter": KalmanFilterSmoother(), 
        "Moving Average": MovingAverageSmoother()
    }
    
    for name, smoother in smoothers.items():
        print(f"{name} Results:")
        smoother.reset()
        
        for frame in test_frames:
            result = smoother.add_frame(frame)
            if result and result.landmarks:
                wrist_pos = result.landmarks.get("wrist", "Missing")
                print(f"  Frame {frame.frame_id}: {wrist_pos} (confidence: {result.confidence:.2f})")
            else:
                print(f"  Frame {frame.frame_id}: No pose detected")
        print()


if __name__ == "__main__":
    compare_smoothing_methods()