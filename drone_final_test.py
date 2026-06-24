import os
import time
import threading
import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────
# VLC path
# ─────────────────────────────────────────────

vlc_paths = [
    r"C:\Program Files\VideoLAN\VLC",
    r"C:\Program Files (x86)\VideoLAN\VLC"
]

for vlc_path in vlc_paths:
    if os.path.exists(vlc_path):
        os.add_dll_directory(vlc_path)
        print("VLC loaded from:", vlc_path)
        break
else:
    print("VLC not found. If video does not work, install VLC 64-bit.")

from flask import Flask, jsonify, Response
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI


# ─────────────────────────────────────────────
# Configuration générale
# ─────────────────────────────────────────────

app = Flask(__name__)

# True  = mode sécurité : les commandes sont reçues mais le drone ne bouge pas
# False = mode réel : le drone peut vraiment bouger
TEST_MODE = False

# Mode actuel :
# "manuel"     = contrôle par Android
# "mouvement"  = contrôle par gestes
CONTROL_MODE = "manuel"

# True = les gestes détectés peuvent envoyer des commandes
# False = les gestes sont seulement affichés, sans contrôle du drone
GESTURE_CONTROL_ENABLED = False

bebop = Bebop(drone_type="Bebop2")

connected = False
video_started = False

latest_frame = None
latest_gesture = "NO_HAND"
latest_stable_gesture = "NO_HAND"

frame_lock = threading.Lock()
command_lock = threading.Lock()
mode_lock = threading.Lock()

# Cooldown général pour les gestes importants : takeoff / land / stop
last_command_time = 0
COMMAND_COOLDOWN = 2.0

# Cooldown spécial pour les mouvements par gestes
# Avant : le drone ne bougeait presque pas car 0.2 s toutes les 2 s
GESTURE_MOVEMENT_COOLDOWN = 0.35
GESTURE_MOVEMENT_DURATION = 0.35

# Anti-accumulation pour Android
last_android_command_time = 0
ANDROID_COMMAND_INTERVAL = 0.15

# Durée courte pour Android, pour éviter que le drone continue trop longtemps
MOVEMENT_DURATION = 0.20

MOVEMENT_COMMANDS = ["forward", "backward", "left", "right", "up", "down"]
SAFETY_COMMANDS = ["land", "stop"]

# Console de sécurité
EMERGENCY_RUNNING = True


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

    # Index, majeur, annulaire, auriculaire
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

def get_movement_duration(source):
    """
    Android 和手势使用不同的移动时间。
    Android 长按会频繁发命令，所以 duration 短。
    手势触发没那么频繁，所以 duration 稍微长一点。
    """
    if source == "gesture":
        return GESTURE_MOVEMENT_DURATION
    return MOVEMENT_DURATION


def execute_drone_command(command, source="unknown"):
    """
    Fonction centrale pour exécuter une commande.
    Utilisée par Android, la reconnaissance de gestes et la console de sécurité.

    Protection :
    - éviter l'accumulation des commandes Android
    - ignorer les commandes de mouvement si une commande est déjà en cours
    - réduire la durée des mouvements Android
    - rendre les mouvements par gestes plus visibles
    - console toujours autorisée pour stop / land
    """

    global connected
    global last_android_command_time

    if not connected:
        return {
            "ok": False,
            "error": "Drone not connected",
            "command": command,
            "source": source,
            "mode": CONTROL_MODE
        }

    with mode_lock:
        current_mode = CONTROL_MODE

    # En mode manuel, les gestes ne contrôlent pas le drone
    if source == "gesture" and current_mode != "mouvement":
        return {
            "ok": False,
            "status": "ignored",
            "reason": "gesture command ignored because current mode is manuel",
            "command": command,
            "source": source,
            "mode": current_mode
        }

    # En mode mouvement, Android ne contrôle pas le drone.
    # Exception sécurité : land et stop restent autorisés depuis Android.
    if source == "android" and current_mode != "manuel":
        if command not in SAFETY_COMMANDS:
            return {
                "ok": False,
                "status": "ignored",
                "reason": "android command ignored because current mode is mouvement",
                "command": command,
                "source": source,
                "mode": current_mode
            }

    # Anti-spam Android : si Android envoie trop vite, on ignore.
    if source == "android" and command in MOVEMENT_COMMANDS:
        now = time.time()
        if now - last_android_command_time < ANDROID_COMMAND_INTERVAL:
            return {
                "ok": False,
                "status": "ignored",
                "reason": "android command too frequent",
                "command": command,
                "source": source,
                "mode": current_mode
            }
        last_android_command_time = now

    if TEST_MODE:
        print(f"[TEST_MODE] {source} command received: {command} | mode={current_mode}")
        return {
            "ok": True,
            "status": "TEST_MODE",
            "command": command,
            "source": source,
            "mode": current_mode,
            "message": "Command received, but drone not controlled"
        }

    # Si une commande de mouvement est déjà en cours,
    # on ignore la nouvelle commande au lieu de la mettre en attente.
    if command in MOVEMENT_COMMANDS:
        locked = command_lock.acquire(blocking=False)

        if not locked:
            return {
                "ok": False,
                "status": "ignored",
                "reason": "another movement command is already running",
                "command": command,
                "source": source,
                "mode": current_mode
            }
    else:
        # takeoff / land / stop attendent le verrou
        command_lock.acquire()

    try:
        print(f"[REAL] {source} command: {command} | mode={current_mode}")

        movement_duration = get_movement_duration(source)

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
                duration=movement_duration
            )

        elif command == "backward":
            bebop.fly_direct(
                roll=0,
                pitch=-10,
                yaw=0,
                vertical_movement=0,
                duration=movement_duration
            )

        elif command == "left":
            bebop.fly_direct(
                roll=-10,
                pitch=0,
                yaw=0,
                vertical_movement=0,
                duration=movement_duration
            )

        elif command == "right":
            bebop.fly_direct(
                roll=10,
                pitch=0,
                yaw=0,
                vertical_movement=0,
                duration=movement_duration
            )

        elif command == "up":
            bebop.fly_direct(
                roll=0,
                pitch=0,
                yaw=0,
                vertical_movement=10,
                duration=movement_duration
            )

        elif command == "down":
            bebop.fly_direct(
                roll=0,
                pitch=0,
                yaw=0,
                vertical_movement=-10,
                duration=movement_duration
            )

        elif command == "stop":
            bebop.fly_direct(
                roll=0,
                pitch=0,
                yaw=0,
                vertical_movement=0,
                duration=0.1
            )

        else:
            return {
                "ok": False,
                "error": "Unknown command",
                "command": command,
                "source": source,
                "mode": current_mode
            }

    finally:
        command_lock.release()

    return {
        "ok": True,
        "status": "executed",
        "command": command,
        "source": source,
        "mode": current_mode
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
    Le contrôle par gestes fonctionne seulement en mode mouvement.
    """

    global last_command_time

    with mode_lock:
        current_mode = CONTROL_MODE
        gesture_enabled = GESTURE_CONTROL_ENABLED

    if current_mode != "mouvement":
        return

    if not gesture_enabled:
        return

    if stable_gesture in ["NO_HAND"]:
        return

    command = command_from_gesture(stable_gesture)

    if command is None:
        return

    now = time.time()

    # Les mouvements doivent être envoyés plus souvent.
    # takeoff / land / stop doivent rester moins fréquents.
    if command in MOVEMENT_COMMANDS:
        cooldown = GESTURE_MOVEMENT_COOLDOWN
    else:
        cooldown = COMMAND_COOLDOWN

    if now - last_command_time < cooldown:
        return

    last_command_time = now

    print("[GESTURE READY]")
    print("Stable gesture:", stable_gesture)
    print("Command:", command)
    print("Mode:", current_mode)
    print("Gesture enabled:", gesture_enabled)

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

            with mode_lock:
                current_mode = CONTROL_MODE

            cv2.putText(
                frame_ai,
                "Mode: " + current_mode,
                (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2
            )

            with frame_lock:
                latest_frame = frame_ai.copy()

            now = time.time()

            if now - last_print_time > 0.5:
                print("Detected:", detected_gesture, "| Stable:", stable_gesture, "| Mode:", current_mode)
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
        "control_mode": CONTROL_MODE,
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
        "control_mode": CONTROL_MODE,
        "latest_gesture": latest_gesture,
        "latest_stable_gesture": latest_stable_gesture,
        "android_movement_duration": MOVEMENT_DURATION,
        "gesture_movement_duration": GESTURE_MOVEMENT_DURATION,
        "gesture_movement_cooldown": GESTURE_MOVEMENT_COOLDOWN,
        "android_command_interval": ANDROID_COMMAND_INTERVAL
    })


@app.route("/manuel", methods=["GET", "POST"])
def set_manuel():
    """
    Mode manuel :
    Android contrôle le drone.
    Les gestes sont détectés et affichés, mais ne contrôlent pas le drone.
    """
    global CONTROL_MODE
    global GESTURE_CONTROL_ENABLED

    with mode_lock:
        CONTROL_MODE = "manuel"
        GESTURE_CONTROL_ENABLED = False

    return jsonify({
        "ok": True,
        "control_mode": CONTROL_MODE,
        "gesture_control_enabled": GESTURE_CONTROL_ENABLED,
        "message": "Mode manuel activated: Android control enabled, gesture control disabled"
    })


@app.route("/mouvement", methods=["GET", "POST"])
def set_mouvement():
    """
    Mode mouvement :
    Les gestes contrôlent le drone.
    Les commandes Android sont ignorées sauf land / stop pour la sécurité.
    """
    global CONTROL_MODE
    global GESTURE_CONTROL_ENABLED

    with mode_lock:
        CONTROL_MODE = "mouvement"
        GESTURE_CONTROL_ENABLED = True

    return jsonify({
        "ok": True,
        "control_mode": CONTROL_MODE,
        "gesture_control_enabled": GESTURE_CONTROL_ENABLED,
        "message": "Mode mouvement activated: gesture control enabled, Android movement commands disabled"
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
    global CONTROL_MODE

    with mode_lock:
        CONTROL_MODE = "mouvement"
        GESTURE_CONTROL_ENABLED = True

    return jsonify({
        "control_mode": CONTROL_MODE,
        "gesture_control_enabled": GESTURE_CONTROL_ENABLED
    })


@app.route("/gesture/off", methods=["GET", "POST"])
def gesture_off():
    global GESTURE_CONTROL_ENABLED
    global CONTROL_MODE

    with mode_lock:
        CONTROL_MODE = "manuel"
        GESTURE_CONTROL_ENABLED = False

    return jsonify({
        "control_mode": CONTROL_MODE,
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
# Console de sécurité
# ─────────────────────────────────────────────

def emergency_console_listener():
    """
    Console de sécurité.
    Pendant que le programme tourne, on peut taper :
    - s / stop   : arrêt immédiat du mouvement
    - land / l   : atterrissage immédiat
    - mode       : afficher le mode actuel
    - manuel     : passer en mode manuel
    - mouvement  : passer en mode mouvement
    """

    global CONTROL_MODE
    global GESTURE_CONTROL_ENABLED

    print("")
    print("Emergency console ready.")
    print("Type 's' or 'stop' to stop the drone.")
    print("Type 'land' or 'l' to land the drone.")
    print("Type 'manuel' or 'mouvement' to change mode.")
    print("Type 'mode' to show current mode.")
    print("")

    while EMERGENCY_RUNNING:
        try:
            cmd = input().strip().lower()

            if cmd in ["s", "stop"]:
                print("[EMERGENCY] STOP command sent from console")
                result = execute_drone_command("stop", source="console")
                print(result)

            elif cmd in ["land", "l"]:
                print("[EMERGENCY] LAND command sent from console")
                result = execute_drone_command("land", source="console")
                print(result)

            elif cmd == "manuel":
                with mode_lock:
                    CONTROL_MODE = "manuel"
                    GESTURE_CONTROL_ENABLED = False

                print("[MODE] manuel activated from console")

            elif cmd == "mouvement":
                with mode_lock:
                    CONTROL_MODE = "mouvement"
                    GESTURE_CONTROL_ENABLED = True

                print("[MODE] mouvement activated from console")

            elif cmd == "mode":
                with mode_lock:
                    current_mode = CONTROL_MODE
                    gesture_enabled = GESTURE_CONTROL_ENABLED

                print("Current mode:", current_mode)
                print("TEST_MODE:", TEST_MODE)
                print("GESTURE_CONTROL_ENABLED:", gesture_enabled)
                print("connected:", connected)
                print("video_started:", video_started)
                print("latest_gesture:", latest_gesture)
                print("latest_stable_gesture:", latest_stable_gesture)

            else:
                print("Unknown console command. Use: s / stop / land / manuel / mouvement / mode")

        except Exception as e:
            print("Emergency console error:", e)
            time.sleep(0.5)


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

    # 1.5 Lancer la console de sécurité
    emergency_thread = threading.Thread(target=emergency_console_listener)
    emergency_thread.daemon = True
    emergency_thread.start()

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
        print("Default mode: manuel")
        print("Android movement duration:", MOVEMENT_DURATION)
        print("Gesture movement duration:", GESTURE_MOVEMENT_DURATION)
        print("Gesture movement cooldown:", GESTURE_MOVEMENT_COOLDOWN)
        print("Android command interval:", ANDROID_COMMAND_INTERVAL)

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