import cv2
import mediapipe as mp
import csv
import time

# --- CONFIG ---
label = "IDLE" # Change this for each gesture you record
output_file = "gesture_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
time.sleep(2)

print(f"Collecting data for: {label}. Ready? (Press 's' to start)")

collecting = False
count = 0

while count < 500:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks and collecting:
        for hand_lms in results.multi_hand_landmarks:
            # Extract only the X and Y of the 21 landmarks (42 features)
            data_row = []
            for lm in hand_lms.landmark:
                data_row.extend([lm.x, lm.y])
            data_row.append(label) # Add the name of the gesture
            
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_row)
            
            count += 1
            cv2.putText(frame, f"Captured: {count}/500", (50, 50), 1, 2, (0, 255, 0), 2)

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1)
    if key == ord('s'): collecting = True
    if key == ord('q'): break

print("Done!")
cap.release()
cv2.destroyAllWindows()