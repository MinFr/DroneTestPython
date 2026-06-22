import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


def count_fingers(hand_landmarks):
    fingers = []

    tips = [8, 12, 16, 20]

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


while True:
    success, frame = cap.read()

    if not success:
        print("Camera not found")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(hand_landmarks)

            if fingers == 1:
                gesture = "TAKEOFF"
            elif fingers == 5:
                gesture = "LAND"
            elif fingers == 2:
                gesture = "FORWARD"
            elif fingers == 3:
                gesture = "LEFT"
            elif fingers == 4:
                gesture = "RIGHT"
            else:
                gesture = "STOP"

    cv2.putText(
        frame,
        gesture,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Gesture Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()