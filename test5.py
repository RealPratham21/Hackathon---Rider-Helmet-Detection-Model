from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO('new_best_5.pt')

# Open webcam
cap = cv2.VideoCapture(0)  # Change to appropriate index if external webcam is used

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Run YOLO model on the frame
    results = model.track(frame, persist=True)  # Track objects persistently
    
    # Plot results on the frame
    annotated_frame = results[0].plot() if results else frame
    
    # Show the frame
    cv2.imshow('YOLO Live Object Tracking', annotated_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
