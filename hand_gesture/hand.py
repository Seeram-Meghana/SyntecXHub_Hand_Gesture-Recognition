import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

def finger_is_open(tip, pip):
    return tip.y < pip.y   # finger is open if tip is above pip

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No Hand"

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            # Fingers
            thumb = lm[4].x > lm[3].x
            index = finger_is_open(lm[8], lm[6])
            middle = finger_is_open(lm[12], lm[10])
            ring = finger_is_open(lm[16], lm[14])
            pinky = finger_is_open(lm[20], lm[18])

            # Gesture logic
            if thumb and not index and not middle and not ring and not pinky:
                gesture = "Thumbs Up 👍 (Volume Up)"
            elif not thumb and not index and not middle and not ring and not pinky:
                gesture = "Fist ✊ (Pause)"
            elif index and middle and ring and pinky:
                gesture = "Open Palm ✋ (Play)"

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Display
    cv2.putText(frame, gesture, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break  # ESC to exit

cap.release()
cv2.destroyAllWindows()
