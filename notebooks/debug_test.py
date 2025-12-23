print("=== DEBUGGING TEST ===")

# Test 1: Basic imports
print("1. Testing imports...")
try:
    import cv2
    print("✓ OpenCV imported")
except Exception as e:
    print(f"✗ OpenCV failed: {e}")

try:
    from ultralytics import YOLO
    print("✓ YOLO imported")
except Exception as e:
    print(f"✗ YOLO failed: {e}")

# Test 2: Model loading
print("\n2. Testing model loading...")
try:
    model = YOLO('../yolovM_badminton_detection.pt')
    print(f"✓ Model loaded: {model.names}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

# Test 3: Webcam
print("\n3. Testing webcam...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Webcam working: frame shape {frame.shape}")
        else:
            print("✗ Cannot read from webcam")
        cap.release()
    else:
        print("✗ Cannot open webcam")
except Exception as e:
    print(f"✗ Webcam test failed: {e}")

print("\n=== TEST COMPLETE ===")