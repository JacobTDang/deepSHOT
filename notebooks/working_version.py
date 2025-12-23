import cv2
import mediapipe as mp
import time
from ultralytics import YOLO
import torch

def main():
    print("=== BADMINTON DETECTION + POSE ===")
    
    # Load model
    print("Loading YOLOv8m model...")
    yolo_model = YOLO('../yolovM_badminton_detection.pt')
    if torch.cuda.is_available():
        yolo_model.to('cuda')
        print(f"✓ Model on GPU: {torch.cuda.get_device_name(0)}")
    
    # MediaPipe setup
    print("Setting up MediaPipe...")
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    # Detection stabilization
    prev_boxes = []
    
    print("✓ Starting detection loop...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame = cv2.flip(frame, 1)  # Mirror
        
        # YOLO detection every frame
        yolo_results = yolo_model(frame, verbose=False, conf=0.5)
        
        # Extract detections
        players = []
        shuttles = []
        rackets = []
        
        for r in yolo_results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Class mapping
                    class_names = {0: 'person', 1: 'racket', 2: 'shuttle'}
                    detected_class = class_names.get(cls, 'unknown')
                    
                    # Filter detections
                    box_area = (x2 - x1) * (y2 - y1)
                    
                    # Debug: print all raw detections
                    if frame_count % 30 == 0:  # Every 30 frames
                        print(f"Raw detection: {detected_class}, conf={conf:.3f}, area={box_area}")
                    
                    if detected_class == 'person' and conf > 0.4 and box_area > 5000:  # Lower thresholds
                        # Stabilize bbox
                        stable_bbox = (x1, y1, x2, y2)
                        if prev_boxes:
                            px1, py1, px2, py2 = prev_boxes[-1]
                            alpha = 0.7
                            stable_bbox = (
                                int(alpha * x1 + (1-alpha) * px1),
                                int(alpha * y1 + (1-alpha) * py1),
                                int(alpha * x2 + (1-alpha) * px2),
                                int(alpha * y2 + (1-alpha) * py2)
                            )
                        prev_boxes.append(stable_bbox)
                        if len(prev_boxes) > 3:
                            prev_boxes.pop(0)
                        
                        players.append({'bbox': stable_bbox, 'conf': conf})
                        
                    elif detected_class == 'shuttle' and conf > 0.3:  # Lower threshold
                        shuttles.append({'bbox': (x1, y1, x2, y2), 'conf': conf})
                        
                    elif detected_class == 'racket' and conf > 0.2:  # Much lower for rackets
                        rackets.append({'bbox': (x1, y1, x2, y2), 'conf': conf})
        
        # Draw YOLO detections
        for player in players:
            x1, y1, x2, y2 = player['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
            cv2.putText(frame, f"Person {player['conf']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        for shuttle in shuttles:
            x1, y1, x2, y2 = shuttle['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
            cv2.putText(frame, f"Shuttle {shuttle['conf']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for racket in rackets:
            x1, y1, x2, y2 = racket['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
            cv2.putText(frame, f"Racket {racket['conf']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Full-frame pose detection (more reliable)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            # Draw pose skeleton
            mp_draw.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        # Info display
        info_text = f"Frame: {frame_count} | Players: {len(players)} | Shuttles: {len(shuttles)} | Rackets: {len(rackets)}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Badminton Analysis', frame)
        
        # Exit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Detection complete")

if __name__ == "__main__":
    main()