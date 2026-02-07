import cv2
import mediapipe as mp
import time
from gestures import get_gesture

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5, 
    max_num_hands=2
)
mp_draw = mp.solutions.drawing_utils

# 2. Setup Camera (Windows 11 Fix)
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
print("Warming up camera...")
time.sleep(2)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("System Active. Show your hand(s)! Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror-like experience
    frame = cv2.flip(frame, 1)
    
    # Process the frame for landmarks
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    current_gestures = []
    
    # 3. Handle Hand Detections
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw the skeleton on the frame
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Get the name of the gesture from our gestures.py file
            gesture = get_gesture(hand_lms)
            current_gestures.append(gesture)
            
    # 4. Final Command Logic
    final_command = "IDLE"
    
    # Check for Dual-Hand Killswitch
    palm_gestures = ["TAKEOFF", "STOP"]
    palms_detected = sum(1 for g in current_gestures if g in palm_gestures)
    
    if palms_detected == 2:
        final_command = "!!! KILLSWITCH (EMERGENCY) !!!"
    elif len(current_gestures) > 0:
        # If one hand is detected, show that hand's command
        final_command = current_gestures[0]

    # 5. UI Overlays
    # Set color to Red for Killswitch, Green for everything else
    color = (0, 0, 255) if "KILLSWITCH" in final_command else (0, 255, 0)
    
    cv2.putText(frame, f"COMMAND: {final_command}", (50, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.imshow("Drone Controller", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()