import os
import time
import threading
import cv2
import numpy as np
import mss

os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")

from flask import Flask, jsonify, Response
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI


app = Flask(__name__)

# True = les commandes ne contrôlent pas le drone
# False = les commandes contrôlent réellement le drone
TEST_MODE = True

bebop = Bebop(drone_type="Bebop2")

connected = False
video_started = False


# 这里是要截取的屏幕区域
# 你需要根据 DroneVisionGUI 窗口的位置调整
SCREEN_REGION = {
    "top": 100,
    "left": 100,
    "width": 800,
    "height": 500
}


# ─────────────────────────────────────────────
# Connexion / état
# ─────────────────────────────────────────────

@app.route("/connect", methods=["GET", "POST"])
def connect():
    return jsonify({
        "connected": connected,
        "video_started": video_started,
        "test_mode_commands": TEST_MODE,
        "message": "Drone connected by main program" if connected else "Drone not connected"
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "connected": connected,
        "video_started": video_started,
        "test_mode_commands": TEST_MODE,
        "video_method": "screen capture from DroneVisionGUI"
    })


def check_connection():
    if not connected:
        return jsonify({"error": "Drone not connected"}), 400
    return None


def test_response(command):
    return jsonify({
        "status": "TEST MODE",
        "command": command,
        "message": "Command received, but drone not controlled"
    })


# ─────────────────────────────────────────────
# Flux vidéo via capture écran
# ─────────────────────────────────────────────

def improve_frame(frame):
    frame = cv2.convertScaleAbs(frame, alpha=1.10, beta=8)
    return frame


@app.route("/video", methods=["GET"])
def video():
    """
    Flux MJPEG pour Android.
    Il capture une zone de l'écran où se trouve la fenêtre DroneVisionGUI.
    """

    def generate():
        with mss.mss() as sct:
            while True:
                screenshot = sct.grab(SCREEN_REGION)

                frame = np.array(screenshot)

                # BGRA -> BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                frame = improve_frame(frame)

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


# ─────────────────────────────────────────────
# Commandes drone
# ─────────────────────────────────────────────

@app.route("/takeoff", methods=["GET", "POST"])
def takeoff():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("takeoff")

    bebop.safe_takeoff(10)

    return jsonify({"status": "takeoff success"})


@app.route("/land", methods=["GET", "POST"])
def land():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("land")

    bebop.safe_land(10)

    return jsonify({"status": "land success"})


@app.route("/forward", methods=["GET", "POST"])
def forward():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("forward")

    bebop.fly_direct(
        roll=0,
        pitch=10,
        yaw=0,
        vertical_movement=0,
        duration=1
    )

    return jsonify({"status": "forward success"})


@app.route("/backward", methods=["GET", "POST"])
def backward():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("backward")

    bebop.fly_direct(
        roll=0,
        pitch=-10,
        yaw=0,
        vertical_movement=0,
        duration=1
    )

    return jsonify({"status": "backward success"})


@app.route("/left", methods=["GET", "POST"])
def left():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("left")

    bebop.fly_direct(
        roll=-10,
        pitch=0,
        yaw=0,
        vertical_movement=0,
        duration=1
    )

    return jsonify({"status": "left success"})


@app.route("/right", methods=["GET", "POST"])
def right():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("right")

    bebop.fly_direct(
        roll=10,
        pitch=0,
        yaw=0,
        vertical_movement=0,
        duration=1
    )

    return jsonify({"status": "right success"})


@app.route("/up", methods=["GET", "POST"])
def up():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("up")

    bebop.fly_direct(
        roll=0,
        pitch=0,
        yaw=0,
        vertical_movement=10,
        duration=1
    )

    return jsonify({"status": "up success"})


@app.route("/down", methods=["GET", "POST"])
def down():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("down")

    bebop.fly_direct(
        roll=0,
        pitch=0,
        yaw=0,
        vertical_movement=-10,
        duration=1
    )

    return jsonify({"status": "down success"})


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

    # 3. Lancer DroneVisionGUI dans le thread principal
    if connected:
        video_started = True

        print("Starting Bebop video stream in main thread")
        print("Move the DroneVisionGUI window into the capture area.")

        try:
            bebopVision = DroneVisionGUI(
                bebop,
                is_bebop=True,
                user_code_to_run=None,
                user_args=None,
                buffer_size=30
            )

            bebopVision.open_video()

        except Exception as e:
            print("Error while starting DroneVisionGUI:", e)

    else:
        print("Could not connect to drone")

        while True:
            time.sleep(1)