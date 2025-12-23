import cv2
from ultralytics import YOLO
import mediapipe as mp

print("=== MINIMAL WORKING TEST ===")

# Load model
print("Loading model...")
model = YOLO('../yolovM_badminton_detection.pt')
print("✓ Model loaded")

# Setup MediaPipe
print("Setting up MediaPipe...")
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✓ MediaPipe initialized")

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)
print("✓ Webcam opened")

frame_count = 0
print("Starting main loop (press 'q' to quit)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    print(f"Processing frame {frame_count}")
    
    # Run YOLO (every 10th frame to avoid hanging)
    if frame_count % 10 == 0:
        try:
            results = model(frame, verbose=False, device='cuda')
            print(f"  YOLO: {len(results)} results")
        except Exception as e:
            print(f"  YOLO error: {e}")
    
    # Run pose detection
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)
        has_pose = pose_results.pose_landmarks is not None
        print(f"  Pose: {has_pose}")
    except Exception as e:
        print(f"  Pose error: {e}")
    
    # Simple display
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Test', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Exit after 100 frames for testing
    if frame_count > 100:
        print("Test completed - 100 frames processed")
        break

cap.release()
cv2.destroyAllWindows()
print("✓ Cleanup complete")