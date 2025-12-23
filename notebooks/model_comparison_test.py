import cv2
from ultralytics import YOLO
import time

def test_model_performance(model_path, model_name):
    print(f"\n=== Testing {model_name} ===")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(0)
    frame_count = 0
    detection_counts = {'person': 0, 'shuttle': 0, 'racket': 0, 'false_pos': 0}
    
    while frame_count < 100:  # Test 100 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model(frame, verbose=False, conf=0.5)
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate metrics
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    aspect_ratio = box_height / box_width if box_width > 0 else 0
                    
                    if model_name == "Custom":
                        class_names = {0: 'person', 1: 'racket', 2: 'shuttle'}
                        detected_class = class_names.get(cls, 'unknown')
                    else:
                        # COCO classes for pretrained
                        if cls == 0:  # person in COCO
                            detected_class = 'person'
                        elif cls == 39:  # sports ball in COCO
                            detected_class = 'shuttle'  
                        elif cls == 38:  # tennis racket in COCO
                            detected_class = 'racket'
                        else:
                            continue
                    
                    # Count detections
                    if detected_class == 'person':
                        if conf > 0.6 and box_area > 3000 and 1.2 < aspect_ratio < 4.0:
                            detection_counts['person'] += 1
                        else:
                            detection_counts['false_pos'] += 1
                            print(f"FALSE POS: conf={conf:.3f}, area={box_area}, ratio={aspect_ratio:.2f}")
                    elif detected_class in ['shuttle', 'racket']:
                        detection_counts[detected_class] += 1
        
        if frame_count % 20 == 0:
            print(f"Frame {frame_count}: {detection_counts}")
    
    cap.release()
    print(f"\nFinal {model_name} results: {detection_counts}")
    false_pos_rate = detection_counts['false_pos'] / max(1, sum(detection_counts.values()))
    print(f"False positive rate: {false_pos_rate:.3f}")
    return detection_counts

if __name__ == "__main__":
    # Test custom model
    custom_results = test_model_performance('../badminton_obj_dectection.pt', 'Custom')
    
    # Test pretrained for comparison
    pretrained_results = test_model_performance('yolov8s.pt', 'Pretrained')
    
    print(f"\n=== COMPARISON ===")
    print(f"Custom false positives: {custom_results['false_pos']}")
    print(f"Pretrained false positives: {pretrained_results['false_pos']}")