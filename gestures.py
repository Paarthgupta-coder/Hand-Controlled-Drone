import math

def get_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_gesture(hand_lms):
    wrist = hand_lms.landmark[0]
    thumb_tip = hand_lms.landmark[4]
    index_tip = hand_lms.landmark[8]
    middle_knuckle = hand_lms.landmark[9] # Stable anchor point
    
    def is_open(tip_idx):
        tip = hand_lms.landmark[tip_idx]
        return get_dist(tip, wrist) > get_dist(hand_lms.landmark[tip_idx - 2], wrist)

    i_open = is_open(8)
    m_open = is_open(12)
    r_open = is_open(16)
    p_open = is_open(20)

    # --- NEW ROBUST THUMB LOGIC ---
    # Thumb UP: Tip is significantly higher (lower Y) than the middle knuckle
    # Thumb DOWN: Tip is significantly lower (higher Y) than the middle knuckle
    thumb_up = thumb_tip.y < middle_knuckle.y - 0.08
    thumb_down = thumb_tip.y > middle_knuckle.y + 0.08

    # 1. LAND (Fist) - Check first
    if not any([i_open, m_open, r_open, p_open]) and not (thumb_up or thumb_down):
        return "LAND"

    # 2. UP / DOWN (Thumb Priority)
    # We only trigger UP/DOWN if the other 4 fingers are closed (fist)
    if not any([i_open, m_open, r_open, p_open]):
        if thumb_up: return "UP"
        if thumb_down: return "DOWN"

    # 3. FORWARD (Sign of the Horns)
    if i_open and p_open and not m_open and not r_open:
        return "FORWARD"

    # 4. BACKWARD (Peace Sign)
    if i_open and m_open and not r_open and not p_open:
        return "BACKWARD"

    # 5. TAKEOFF / STOP (Palm Logic)
    if all([i_open, m_open, r_open, p_open]):
        spread = get_dist(index_tip, hand_lms.landmark[20])
        return "TAKEOFF" if spread > 0.12 else "STOP"

    return "IDLE"