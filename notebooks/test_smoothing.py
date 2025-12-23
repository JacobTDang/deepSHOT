import cv2
import mediapipe as mp
import collections
import time
import math
from smooth_pose_demo import SimplePoseHistory
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", 'src'))
from utils.temporal_smoothing import (PoseFrame, KalmanFilterSmoother, MovingAverageSmoother)
from ultralytics import YOLO

def mediapipe_to_pose_frame(mp_landmarks, timestamp, frame_id, confidence):
  # convert mediapipe landmarks to poseframe format
  if mp_landmarks is None:
    return PoseFrame(timestamp, None, confidence, frame_id)

  landmarks={}
  for i, landmark in enumerate(mp_landmarks.landmark):
    landmarks[f'point_{i}'] = (landmark.x, landmark.y)
  return PoseFrame(timestamp, landmarks, confidence, frame_id)

def calculate_joint_jitter(pose_history, joint_name):
    """Calculate frame-to-frame distance for specific joint"""
    distances = []
    valid_frames = [f for f in pose_history if f.get('landmarks')]

    for i in range(1, len(valid_frames)):
        prev_frame = valid_frames[i-1]
        curr_frame = valid_frames[i]

        # Extract joint positions (for MediaPipe landmarks)
        if (prev_frame['landmarks'] and curr_frame['landmarks'] and
            hasattr(prev_frame['landmarks'], 'landmark') and
            hasattr(curr_frame['landmarks'], 'landmark')):

            joint_idx = int(joint_name.split('_')[1]) if 'point_' in joint_name else 0

            if (len(prev_frame['landmarks'].landmark) > joint_idx and
                len(curr_frame['landmarks'].landmark) > joint_idx):

                prev_pos = prev_frame['landmarks'].landmark[joint_idx]
                curr_pos = curr_frame['landmarks'].landmark[joint_idx]

                dist = math.sqrt((curr_pos.x - prev_pos.x)**2 + (curr_pos.y - prev_pos.y)**2)
                distances.append(dist)

    return sum(distances) / len(distances) if distances else 0

def calculate_pose_frame_jitter(pose_frame_history, joint_name):
    """Calculate jitter for PoseFrame format (for advanced smoothers)"""
    distances = []
    valid_frames = [f for f in pose_frame_history if f and f.landmarks]

    for i in range(1, len(valid_frames)):
        prev_frame = valid_frames[i-1]
        curr_frame = valid_frames[i]

        if (joint_name in prev_frame.landmarks and joint_name in curr_frame.landmarks):
            prev_x, prev_y = prev_frame.landmarks[joint_name]
            curr_x, curr_y = curr_frame.landmarks[joint_name]

            dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            distances.append(dist)

    return sum(distances) / len(distances) if distances else 0



def main():
  # Load  trained multi-class YOLO model
  import torch
  # Use your trained YOLOv8m custom model
  print("Loading YOLOv8m model...")
  try:
    yolo_model = YOLO('../yolovM_badminton_detection.pt')
    print("✓ YOLOv8m model loaded successfully")
  except Exception as e:
    print(f"✗ Error loading model: {e}")
    return

  # Force YOLO to GPU, MediaPipe will use CPU
  if torch.cuda.is_available():
    print("Moving model to GPU...")
    yolo_model.to('cuda')
    print(f"✓ YOLO model loaded on GPU: {torch.cuda.get_device_name(0)}")
    # Clear any GPU memory fragmentation
    torch.cuda.empty_cache()
  else:
    print("✓ YOLO model loaded on CPU")

  # MediaPipe setup - configure for CPU-only
  mp_pose = mp.solutions.pose
  mp_draw = mp.solutions.drawing_utils

  # Initialize smoothers for each player
  player1_smoothers = {
    'simple': SimplePoseHistory(5),
    'ma5': MovingAverageSmoother(5)
  }

  player2_smoothers = {
    'simple': SimplePoseHistory(5),
    'ma5': MovingAverageSmoother(5)
  }

  # Data storage for jitter analysis
  raw_pose_history = []
  ma3_history = []
  ma5_history = []
  ma10_history = []

  # Detection stabilization - track previous boxes
  prev_player_boxes = []

  # Key joints for badminton analysis
  key_joints = ['point_15', 'point_16']  # Left/right wrists in MediaPipe

  # Configure MediaPipe for CPU-only processing
  pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # Use lighter model to reduce CPU load
    smooth_landmarks=False,  # We'll handle smoothing ourselves
    min_detection_confidence=0.5,  # Lower for better detection
    min_tracking_confidence=0.5   # Lower for better tracking
  )

  # webcam setup
  print("Initializing webcam...")
  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    print("✗ Error: Cannot open webcam!")
    return

  # Test if we can read a frame
  ret, test_frame = cap.read()
  if not ret:
    print("✗ Error: Cannot read from webcam!")
    return

  print(f"✓ Webcam initialized: {test_frame.shape}")
  frame_count=0

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    frame_count +=1
    frame = cv2.flip(frame,1)

    # Run YOLO detection with explicit GPU usage
    with torch.cuda.device(0):  # Force GPU context
      yolo_results = yolo_model(frame, verbose=False, conf=0.5, iou=0.7, device='cuda')

    # Extract detected objects
    players = []
    shuttles = []
    rackets = []

    # Debug: Check if YOLO returns anything at all
    print(f"YOLO results count: {len(yolo_results)}")

    for r in yolo_results:
      boxes = r.boxes
      print(f"Boxes in result: {boxes is not None and len(boxes) if boxes is not None else 'None'}")
      if boxes is not None:
        for box in boxes:
          cls = int(box.cls[0])
          conf = float(box.conf[0])
          x1, y1, x2, y2 = map(int, box.xyxy[0])

          # Calculate box dimensions
          box_width = x2 - x1
          box_height = y2 - y1
          box_area = box_width * box_height

          # Debug: Print ALL detections before any filtering
          print(f"RAW YOLO: class={cls}, conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2}), size={box_width}x{box_height}, area={box_area}")

          # Your custom badminton model classes
          class_names = {0: 'person', 1: 'racket', 2: 'shuttle'}
          detected_class = class_names.get(cls, 'unknown')

          if detected_class == 'unknown':
            print(f"  -> Unknown class {cls}: conf={conf:.3f}")
            continue

          # Relaxed filtering for person detection
          if detected_class == 'person' and conf > 0.7 and box_area > 15000:
            # More relaxed aspect ratio for close-up/sitting people
            aspect_ratio = box_height / box_width if box_width > 0 else 0
            if 0.6 < aspect_ratio < 3.0:  # Allow wider people (sitting/close)
              # Stabilize box coordinates
              stable_bbox = (x1, y1, x2, y2)
              if prev_player_boxes:
                # Use moving average for bbox stabilization
                prev_x1, prev_y1, prev_x2, prev_y2 = prev_player_boxes[-1]
                alpha = 0.7  # Smoothing factor
                stable_x1 = int(alpha * x1 + (1-alpha) * prev_x1)
                stable_y1 = int(alpha * y1 + (1-alpha) * prev_y1)
                stable_x2 = int(alpha * x2 + (1-alpha) * prev_x2)
                stable_y2 = int(alpha * y2 + (1-alpha) * prev_y2)
                stable_bbox = (stable_x1, stable_y1, stable_x2, stable_y2)

              players.append({'class': 'person', 'bbox': stable_bbox, 'conf': conf})
              prev_player_boxes.append(stable_bbox)
              if len(prev_player_boxes) > 5:  # Keep only recent boxes
                prev_player_boxes.pop(0)
              print(f"  -> Valid person: conf={conf:.3f}, area={box_area}, ratio={aspect_ratio:.2f}")
            else:
              print(f"  -> Filtered person (bad ratio): ratio={aspect_ratio:.2f}")
          elif detected_class == 'shuttle' and conf > 0.4 and box_area > 30:
            shuttles.append({'bbox': (x1, y1, x2, y2), 'conf': conf})
            print(f"  -> Valid shuttle: conf={conf:.3f}, area={box_area}")
          elif detected_class == 'racket' and conf > 0.4 and box_area > 400:
            rackets.append({'bbox': (x1, y1, x2, y2), 'conf': conf})
            print(f"  -> Valid racket: conf={conf:.3f}, area={box_area}")
          else:
            print(f"  -> FILTERED: {detected_class} conf={conf:.3f}, area={box_area}")

    # Process pose for each detected player
    player_poses = {}
    for player in players:
      player_id = player['class']
      x1, y1, x2, y2 = player['bbox']

      # Crop player region and process on CPU
      player_crop = frame[y1:y2, x1:x2]
      if player_crop.size > 0:
        rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        # Ensure MediaPipe uses CPU
        pose_results = pose.process(rgb_crop)

        if pose_results.pose_landmarks:
          # Adjust landmarks back to full frame coordinates
          adjusted_landmarks = pose_results.pose_landmarks
          for landmark in adjusted_landmarks.landmark:
            landmark.x = (landmark.x * (x2 - x1) + x1) / frame.shape[1]
            landmark.y = (landmark.y * (y2 - y1) + y1) / frame.shape[0]

          confidences = [lm.visibility for lm in adjusted_landmarks.landmark]
          confidence = sum(confidences) / len(confidences)
          player_poses[player_id] = {'landmarks': adjusted_landmarks, 'confidence': confidence}

    # Process smoothing for each detected player
    timestamp = time.time()

    # Draw YOLO detections
    for player in players:
      x1, y1, x2, y2 = player['bbox']
      cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow boxes
      cv2.putText(frame, f"{player['class']}: {player['conf']:.2f}",
                 (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    for shuttle in shuttles:
      x1, y1, x2, y2 = shuttle['bbox']
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Cyan boxes
      cv2.putText(frame, f"shuttle: {shuttle['conf']:.2f}",
                 (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Process pose smoothing for each player
    for player_id, pose_data in player_poses.items():
      landmarks = pose_data['landmarks']
      confidence = pose_data['confidence']

      # Convert to PoseFrame format
      pose_frame = mediapipe_to_pose_frame(landmarks, timestamp, frame_count, confidence)

      # Select appropriate smoother set
      if player_id == 'player1':
        smoothers = player1_smoothers
      else:
        smoothers = player2_smoothers

      # Apply smoothing
      smoothers['simple'].add_frame(landmarks, confidence, frame_count)
      smoothed_result = smoothers['ma5'].add_frame(pose_frame)

      # Draw poses for this player
      if landmarks:
        mp_draw.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                             landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                             connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2))

    # Skip backup pose - YOLO is reliable enough
    # (Backup pose was causing confusion and visual clutter)

    # Display detection info
    detection_info = f"Players: {len(players)}, Shuttles: {len(shuttles)}, Rackets: {len(rackets)}"
    cv2.putText(frame, detection_info, (10, frame.shape[0] - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Helper function to convert PoseFrame back to MediaPipe format
    def pose_frame_to_mediapipe(pose_frame):
      if not pose_frame or not pose_frame.landmarks:
        return None

      # Create a simple object that mimics MediaPipe landmarks structure
      class MockLandmarks:
        def __init__(self, landmarks_dict):
          self.landmark = []
          # Convert back to MediaPipe format (assuming 33 pose landmarks)
          for i in range(33):  # MediaPipe has 33 pose landmarks
            point_name = f"point_{i}"
            if point_name in landmarks_dict:
              x, y = landmarks_dict[point_name]
              # Create mock landmark object
              mock_landmark = type('MockLandmark', (), {'x': x, 'y': y, 'z': 0, 'visibility': 0.9})()
              self.landmark.append(mock_landmark)
            else:
              # Default landmark if missing
              mock_landmark = type('MockLandmark', (), {'x': 0, 'y': 0, 'z': 0, 'visibility': 0})()
              self.landmark.append(mock_landmark)

      return MockLandmarks(pose_frame.landmarks)

    # Add detection summary
    cv2.putText(frame, "YOLO + Pose Detection", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the frame
    try:
      cv2.imshow('Pose Smoothing', frame)
      print(f"Frame {frame_count} displayed") if frame_count % 30 == 0 else None
    except Exception as e:
      print(f"Error displaying frame: {e}")
      break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      print("Quitting...")
      break

  # Cleanup
  print("Cleaning up...")
  cap.release()
  cv2.destroyAllWindows()
  print("Done")


if __name__ == "__main__":
  main()
