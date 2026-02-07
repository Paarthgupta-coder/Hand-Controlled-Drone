import cv2
import mediapipe as mp
import time
import sys

print("1. Initializing MediaPipe...")
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    print("   Success!")
except Exception as e:
    print(f"   FAILED to start MediaPipe: {e}")
    sys.exit()

print("2. Starting Camera (MSMF)...")
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

print("3. Warming up for 2 seconds...")
time.sleep(2)

if not cap.isOpened():
    print("   ERROR: Camera hardware not found or busy.")
    sys.exit()

print("4. Entering main loop. Press 'q' to quit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("   Failed to grab frame.")
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Drone Gesture Control", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"   CRASH in loop: {e}")
finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()