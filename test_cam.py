import cv2
import time

# We use CAP_MSMF for Windows 11 compatibility
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

# Give the camera a second to warm up (prevents the 'fraction of a sec' blink)
print("Warming up camera...")
time.sleep(2)

if not cap.isOpened():
    print("Error: Could not open video device.")
else:
    print("Camera opened successfully! Press 'q' to close the window.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        cv2.imshow('Drone Cam Test', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()