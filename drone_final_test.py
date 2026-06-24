import os
import time
import threading
import cv2
import mediapipe as mp
import numpy as np

os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")

from flask import Flask, jsonify, Response
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI


# ─────────────────────────────────────────────
# Configuration générale
# ─────────────────────────────────────────────

app = Flask(__name__)

# True  = mode sécurité : les commandes sont reçues mais le drone ne bouge pas
# False = mode réel : le drone peut vraiment bouger
TEST_MODE = True

# True = les gestes détectés peuvent envoyer des commandes
# False = les gestes sont seulement affichés, sans contrôle du drone
GESTURE_CONTROL_ENABLED = True

bebop = Bebop(drone_type="Bebop2")

connected = False
video_started = False

latest_frame = None
latest_gesture = "NO_HAND"
latest_stable_gesture = "NO_HAND"

frame_lock = threading.Lock()
command_lock = threading.Lock()

last_command_time = 0
COMMAND_COOLDOWN = 2.0  # secondes entre deux commandes de geste


# ─────────────────────────────────────────────
# MediaPipe Hands
# ─────────────────────────────────────────────

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)


# ─────────────────────────────────────────────
# Outils image
# ─────────────────────────────────────────────

def improve_frame_quality(frame):
    """
    Amélioration visuelle simple :
    contraste, luminosité, léger flou, netteté.
    """
    frame = cv2.convertScaleAbs(frame, alpha=1.20, beta=12)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    frame = cv2.filter2D(frame, -1, sharpen_kernel)

    return frame


def waiting_frame(text="Waiting for real drone camera..."):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        text,
        (60, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        time.strftime("%H:%M:%S"),
        (240, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    return frame


# ─────────────────────────────────────────────
# Détection des doigts
# ─────────────────────────────────────────────

def count_fingers(hand_landmarks, handedness_label):
    """
    Compte les doigts levés.
    Index, majeur, annulaire, auriculaire : comparaison y.
    Pouce : comparaison x selon main droite / main gauche.
    """

    fingers = []

    # Pouce
    # Attention : selon la caméra, l'effet miroir peut inverser gauche/droite.
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    if handedness_label == "Right":
        if thumb_tip.x < thumb_ip.x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if thumb_tip.x > thumb_ip.x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Autres doigts : index, majeur, annulaire, auriculaire
    tips = [8, 12, 16, 20]

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


def detect_gesture_and_draw(frame):
    """
    Détecte le geste et dessine les points de la main.
    """

    frame = improve_frame_quality(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NO_HAND"

    if result.multi_hand_landmarks:
        for index, hand_landmarks in enumerate(result.multi_hand_landmarks):

            handedness_label = "Right"
            if result.multi_handedness:
                handedness_label = result.multi_handedness[index].classification[0].label

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            fingers = count_fingers(hand_landmarks, handedness_label)

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
                f"Fingers: {fingers}",
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

    cv2.putText(
        frame,
        "Detected: " + gesture,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    return frame, gesture


# ─────────────────────────────────────────────
# Commandes drone communes
# ─────────────────────────────────────────────

def execute_drone_command(command, source="unknown"):
    """
    Fonction centrale pour exécuter une commande.
    Utilisée par Android et par la reconnaissance de gestes.
    """

    global connected

    if not connected:
        return {
            "ok": False,
            "error": "Drone not connected",
            "command": command,
            "source": source
        }

    if TEST_MODE:
        print(f"[TEST_MODE] {source} command received: {command}")
        return {
            "ok": True,
            "status": "TEST_MODE",
            "command": command,
            "source": source,
            "message": "Command received, but drone not controlled"
        }

    with command_lock:
        print(f"[REAL] {source} command: {command}")

        if command == "takeoff":
            bebop.safe_takeoff(10)

        elif command == "land":
            bebop.safe_land(10)

        elif command == "forward":
            bebop.fly_direct(
                roll=0,
                pitch=10,
                yaw=0,
                vertical_movement=0,
                duration=1
            )

        elif command == "backward":
            bebop.fly_direct(
                roll=0,
                pitch=-10,
                yaw=0,
                vertical_movement=0,
                duration=1
            )

        elif command == "left":
            bebop.fly_direct(
                roll=-10,
                pitch=0,
                yaw=0,
                vertical_movement=0,
                duration=1
            )

        elif command == "right":
            bebop.fly_direct(
                roll=10,
                pitch=0,
                yaw=0,
                vertical_movement=0,
                duration=1
            )

        elif command == "up":
            bebop.fly_direct(
                roll=0,
                pitch=0,
                yaw=0,
                vertical_movement=10,
                duration=1
            )

        elif command == "down":
            bebop.fly_direct(
                roll=0,
                pitch=0,
                yaw=0,
                vertical_movement=-10,
                duration=1
            )

        elif command == "stop":
            bebop.fly_direct(
                roll=0,
                pitch=0,
                yaw=0,
                vertical_movement=0,
                duration=1
            )

        else:
            return {
                "ok": False,
                "error": "Unknown command",
                "command": command,
                "source": source
            }

    return {
        "ok": True,
        "status": "executed",
        "command": command,
        "source": source
    }


def command_from_gesture(gesture):
    """
    Association geste -> commande.
    """
    mapping = {
        "TAKEOFF": "takeoff",
        "FORWARD": "forward",
        "LEFT": "left",
        "RIGHT": "right",
        "LAND": "land",
        "STOP": "stop"
    }

    return mapping.get(gesture)


def handle_stable_gesture(stable_gesture):
    """
    Envoie une commande quand le geste est stable.
    Ajoute un cooldown pour éviter d'envoyer 20 commandes par seconde.
    """

    global last_command_time

    if not GESTURE_CONTROL_ENABLED:
        return

    command = command_from_gesture(stable_gesture)

    if command is None:
        return

    now = time.time()

    if now - last_command_time < COMMAND_COOLDOWN:
        return

    # On évite de répéter STOP tout le temps
    if stable_gesture in ["NO_HAND"]:
        return

    last_command_time = now

    result = execute_drone_command(command, source="gesture")
    print("Gesture command result:", result)


# ─────────────────────────────────────────────
# Code vidéo + IA
# ─────────────────────────────────────────────

def user_code(drone_vision, user_args):
    """
    Fonction appelée par DroneVisionGUI.
    Elle récupère les images du Bebop, détecte les gestes,
    affiche la vidéo IA, et déclenche les commandes si besoin.
    """

    global latest_frame
    global latest_gesture
    global latest_stable_gesture

    last_print_time = 0

    candidate_gesture = ""
    candidate_count = 0
    stable_gesture = "NO_HAND"

    STABLE_FRAME_LIMIT = 5

    print("Commencer AI gesture detection")

    while True:
        try:
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

            latest_gesture = detected_gesture
            latest_stable_gesture = stable_gesture

            # Si le geste est stable, on peut envoyer une commande
            handle_stable_gesture(stable_gesture)

            cv2.putText(
                frame_ai,
                "Stable: " + stable_gesture,
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 0, 0),
                3
            )

            cv2.putText(
                frame_ai,
                "TEST_MODE: " + str(TEST_MODE),
                (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2
            )

            cv2.putText(
                frame_ai,
                "Gesture control: " + str(GESTURE_CONTROL_ENABLED),
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2
            )

            with frame_lock:
                latest_frame = frame_ai.copy()

            now = time.time()

            if now - last_print_time > 0.5:
                print("Detected:", detected_gesture, "| Stable:", stable_gesture)
                last_print_time = now

            cv2.imshow("Drone Camera AI Gesture Detection", frame_ai)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.03)

        except Exception as e:
            print("Error in user_code:", e)
            time.sleep(1)

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Routes Flask pour Android
# ─────────────────────────────────────────────

@app.route("/connect", methods=["GET", "POST"])
def connect():
    return jsonify({
        "connected": connected,
        "video_started": video_started,
        "has_frame": latest_frame is not None,
        "test_mode_commands": TEST_MODE,
        "gesture_control_enabled": GESTURE_CONTROL_ENABLED,
        "message": "Drone connected by main program" if connected else "Drone not connected"
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "connected": connected,
        "video_started": video_started,
        "has_frame": latest_frame is not None,
        "test_mode_commands": TEST_MODE,
        "gesture_control_enabled": GESTURE_CONTROL_ENABLED,
        "latest_gesture": latest_gesture,
        "latest_stable_gesture": latest_stable_gesture
    })


@app.route("/video", methods=["GET"])
def video():
    """
    Flux MJPEG pour Android :
    http://IP_DU_PC:5000/video
    """

    def generate():
        while True:
            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()

            if frame is None:
                frame = waiting_frame("Waiting for real drone camera...")

            success, buffer = cv2.imencode(".jpg", frame)

            if success:
                jpg = buffer.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/takeoff", methods=["GET", "POST"])
def takeoff():
    result = execute_drone_command("takeoff", source="android")
    return jsonify(result)


@app.route("/land", methods=["GET", "POST"])
def land():
    result = execute_drone_command("land", source="android")
    return jsonify(result)


@app.route("/forward", methods=["GET", "POST"])
def forward():
    result = execute_drone_command("forward", source="android")
    return jsonify(result)


@app.route("/backward", methods=["GET", "POST"])
def backward():
    result = execute_drone_command("backward", source="android")
    return jsonify(result)


@app.route("/left", methods=["GET", "POST"])
def left():
    result = execute_drone_command("left", source="android")
    return jsonify(result)


@app.route("/right", methods=["GET", "POST"])
def right():
    result = execute_drone_command("right", source="android")
    return jsonify(result)


@app.route("/up", methods=["GET", "POST"])
def up():
    result = execute_drone_command("up", source="android")
    return jsonify(result)


@app.route("/down", methods=["GET", "POST"])
def down():
    result = execute_drone_command("down", source="android")
    return jsonify(result)


@app.route("/stop", methods=["GET", "POST"])
def stop():
    result = execute_drone_command("stop", source="android")
    return jsonify(result)


@app.route("/gesture/on", methods=["GET", "POST"])
def gesture_on():
    global GESTURE_CONTROL_ENABLED
    GESTURE_CONTROL_ENABLED = True

    return jsonify({
        "gesture_control_enabled": GESTURE_CONTROL_ENABLED
    })


@app.route("/gesture/off", methods=["GET", "POST"])
def gesture_off():
    global GESTURE_CONTROL_ENABLED
    GESTURE_CONTROL_ENABLED = False

    return jsonify({
        "gesture_control_enabled": GESTURE_CONTROL_ENABLED
    })


@app.route("/disconnect", methods=["GET", "POST"])
def disconnect():
    global connected

    if connected:
        bebop.disconnect()
        connected = False

    return jsonify({
        "status": "disconnected"
    })


# ─────────────────────────────────────────────
# Flask en arrière-plan
# ─────────────────────────────────────────────

def run_flask():
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        use_reloader=False
    )


# ─────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Lancer Flask dans un thread secondaire
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # 2. Connexion réelle au Bebop
    print("Connecting to Bebop...")
    connected = bebop.connect(10)
    print("Connection:", connected)

    # 3. Essayer de rendre le flux vidéo plus stable
    if connected:
        try:
            if hasattr(bebop, "set_video_stream_mode"):
                bebop.set_video_stream_mode("high_reliability")
                print("Video stream mode: high reliability")
            else:
                print("set_video_stream_mode not available")
        except Exception as e:
            print("Could not set video stream mode:", e)

        video_started = True

        print("Starting drone camera with AI gesture detection")

        try:
            bebopVision = DroneVisionGUI(
                bebop,
                is_bebop=True,
                user_code_to_run=user_code,
                user_args=None,
                buffer_size=50
            )

            bebopVision.open_video()

        except Exception as e:
            print("Error while starting DroneVisionGUI:", e)

    else:
        print("Could not connect to drone")

        while True:
            time.sleep(1)