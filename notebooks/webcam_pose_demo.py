"""
Live Webcam Pose Detection Demo

This will show you MediaPipe in action - you'll see your skeleton tracked in real-time!
Perfect for understanding how our badminton system will work.

Controls:
- Press 'q' to quit
- Press 's' to save a frame with pose data
"""
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

def main():
    # Initialize pose detector
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=fastest, 1=balanced, 2=best accuracy
        smooth_landmarks=True,  # Smooth tracking over time
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    print("=== Live Pose Detection ===")
    print("Controls:")
    print("- 'q' to quit")
    print("- 's' to save current frame")
    print("- Move around to see pose detection!")
    print("\nStarting webcam...")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame_count += 1
        
        # Flip frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose detection
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            # Draw the pose skeleton
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style()
            )
            
            # Extract key points for learning
            landmarks = results.pose_landmarks.landmark
            
            # Show some key joint coordinates (learning example)
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            # Convert normalized coordinates to pixels
            h, w, c = frame.shape
            left_wrist_px = (int(left_wrist.x * w), int(left_wrist.y * h))
            right_wrist_px = (int(right_wrist.x * w), int(right_wrist.y * h))
            
            # Draw custom annotations
            cv2.circle(frame, left_wrist_px, 10, (0, 255, 0), -1)
            cv2.circle(frame, right_wrist_px, 10, (255, 0, 0), -1)
            
            # Show confidence score
            cv2.putText(frame, f"Detection Confidence: {left_wrist.visibility:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate and display FPS
        if frame_count % 30 == 0:  # Update every 30 frames
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.1f} | Pose detected: {results.pose_landmarks is not None}")
        
        # Add text overlay
        cv2.putText(frame, "DeepShot Pose Demo - Press 'q' to quit, 's' to save", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow('Badminton Pose Analysis Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save frame for analysis
            save_path = Path("data/processed") / f"pose_frame_{frame_count}.jpg"
            save_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(save_path), frame)
            print(f"Saved frame to {save_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    
    print(f"\nDemo complete!")
    print(f"Processed {frame_count} frames in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()