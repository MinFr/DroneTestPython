import os
import time
import cv2
import mediapipe as mp

os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")

from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)


# Amélioration simple de l'image
def improve_frame_quality(frame):
    # 1. Améliorer contraste et luminosité
    # alpha = contraste, beta = luminosité
    frame = cv2.convertScaleAbs(frame, alpha=1.25, beta=15)

    # 2. Réduction légère du bruit
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # 3. Sharpening : rendre l'image un peu plus nette
    sharpen_kernel = (
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    )

    kernel = cv2.UMat(
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    # Méthode plus simple avec un vrai kernel numpy
    import numpy as np
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    frame = cv2.filter2D(frame, -1, sharpen_kernel)

    return frame


def count_fingers(hand_landmarks):
    fingers = []

    # Index, majeur, annulaire, auriculaire
    tips = [8, 12, 16, 20]

    for tip in tips:
        # Dans OpenCV, plus y est petit, plus le point est haut dans l'image
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


def detect_gesture_and_draw(frame):
    # Améliorer l'image avant l'affichage et la détection
    frame = improve_frame_quality(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NO_HAND"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = count_fingers(hand_landmarks)

            if fingers == 1:
                gesture = "TAKEOFF"
            elif fingers == 2:
                gesture = "FORWARD"
            elif fingers == 3:
                gesture = "LEFT"
            elif fingers == 4:
                gesture = "RIGHT"
            elif fingers == 5:
                gesture = "LAND"
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

    return frame, gesture


def user_code(drone_vision, user_args):
    last_print_time = 0

    candidate_gesture = ""
    candidate_count = 0
    stable_gesture = "NO_HAND"

    STABLE_FRAME_LIMIT = 5

    print("Commencer AI gesture detection")

    while True:
        frame = drone_vision.get_latest_valid_picture()

        if frame is None:
            time.sleep(0.1)
            continue

        frame_ai, detected_gesture = detect_gesture_and_draw(frame)

        if detected_gesture == candidate_gesture:
            candidate_count += 1
        else:
            candidate_gesture = detected_gesture
            candidate_count = 1

        if candidate_count >= STABLE_FRAME_LIMIT:
            stable_gesture = candidate_gesture

        cv2.putText(
            frame_ai,
            "Stable: " + stable_gesture,
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3
        )

        now = time.time()

        if now - last_print_time > 0.5:
            print("Detected:", detected_gesture, "| Stable:", stable_gesture)
            last_print_time = now

        cv2.imshow("Drone Camera AI Gesture Detection", frame_ai)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.03)

    cv2.destroyAllWindows()


bebop = Bebop(drone_type="Bebop2")

success = bebop.connect(10)
print("Connection:", success)

if success:
    print("Starting drone camera with AI gesture detection")

    # Essayer de rendre le flux vidéo plus stable si la fonction existe
    try:
        if hasattr(bebop, "set_video_stream_mode"):
            bebop.set_video_stream_mode("high_reliability")
            print("Video stream mode: high reliability")
        else:
            print("set_video_stream_mode not available")
    except Exception as e:
        print("Could not set video stream mode:", e)

    bebopVision = DroneVisionGUI(
        bebop,
        is_bebop=True,
        user_code_to_run=user_code,
        user_args=None,
        buffer_size=50
    )

    bebopVision.open_video()

else:
    print("Could not connect.")