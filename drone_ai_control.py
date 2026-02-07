import cv2
import mediapipe as mp
import pickle
import numpy as np
import time

# 1. Load the AI Model (The 'Brain')
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# 3. Camera Setup
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
time.sleep(2)

print("AI Controller Reverted (Direct Prediction). Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    current_gestures = []

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # EXTRACT FEATURES
            data_row = []
            for lm in hand_lms.landmark:
                data_row.extend([lm.x, lm.y])
            
            # DIRECT PREDICTION (No confidence barrier)
            prediction = model.predict([data_row])
            gesture_name = prediction[0]
            current_gestures.append(gesture_name)

    # 4. Final Logic
    final_command = "IDLE"
    
    # Killswitch (Two hands doing any palm gesture)
    palms = sum(1 for g in current_gestures if g in ["TAKEOFF", "STOP"])
    
    if palms == 2:
        final_command = "!!! KILLSWITCH !!!"
    elif len(current_gestures) > 0:
        # Show the first hand's command immediately
        final_command = current_gestures[0]

    # 5. UI Display
    color = (0, 0, 255) if "KILL" in final_command else (0, 255, 0)
    cv2.putText(frame, f"AI CMD: {final_command}", (50, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.imshow("Drone AI Controller (Reverted)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()