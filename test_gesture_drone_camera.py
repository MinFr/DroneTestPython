import os
import time
import cv2
import mediapipe as mp

os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")

from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI

mp_hands = mp.solutions.hands #solutions pour la détection des mains
mp_draw = mp.solutions.drawing_utils # utilitaire pour dessiner les points de repère sur l'image

# Définir les paramètres pour la détection des mains
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)


def count_fingers(hand_landmarks):
    fingers = []
    
    tips = [8, 12, 16, 20] # les indices des points de repère des doigts (index, majeur, annulaire, auriculaire)

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y: # si le point de repère est plus haut que le point de repère précédent
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


def detect_gesture_and_draw(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb) #transformer l'image en RGB et la traiter pour détecter les mains

    gesture = "NO_HAND"

    if result.multi_hand_landmarks: # si des mains sont détectées
        for hand_landmarks in result.multi_hand_landmarks:
            
            mp_draw.draw_landmarks( #desinner les points de repère sur l'image
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = count_fingers(hand_landmarks) # compter le nombre de doigts levés

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

    cv2.putText( # afficher le geste texte détecté sur l'image
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

    # gesture detection variables
    candidate_gesture = ""

    # nombre de frames consécutifs pour confirmer le geste
    candidate_count = 0

    # gesture stable variable
    stable_gesture = "NO_HAND"

    # nombre de frames consécutifs pour confirmer le geste
    STABLE_FRAME_LIMIT = 5

    print(" Commencer AI gesture detection")

    while True:
        frame = drone_vision.get_latest_valid_picture() # récupérer la dernière image valide de la caméra du drone

        if frame is None:
            time.sleep(0.1)
            continue

        frame_ai, detected_gesture = detect_gesture_and_draw(frame) # détecter le geste et dessiner les points de repère sur l'image

        # si le geste détecté est le même que le candidat précédent, on incrémente le compteur
        if detected_gesture == candidate_gesture:
            candidate_count += 1
        else: # si le geste détecté est différent, on réinitialise le compteur et on met à jour le candidat
            candidate_gesture = detected_gesture
            candidate_count = 1

        # si le compteur atteint la limite, on met à jour le geste stable
        if candidate_count >= STABLE_FRAME_LIMIT:
            stable_gesture = candidate_gesture

        # Afficher le geste stable sur l'image
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

        # Afficher le geste détecté et le geste stable toutes les 0.5 secondes
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

    bebopVision = DroneVisionGUI(
        bebop,
        is_bebop=True,
        user_code_to_run=user_code,
        user_args=None,
        buffer_size=200
    )

    bebopVision.open_video()

else:
    print("Could not connect.")