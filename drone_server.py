from flask import Flask, jsonify
from pyparrot.Bebop import Bebop

app = Flask(__name__)

TEST_MODE = True   # true pour le mode test, false pour le mode réel

bebop = Bebop(drone_type="Bebop2")
connected = False


@app.route("/connect", methods=["GET", "POST"])
def connect():
    global connected

    if TEST_MODE:
        connected = True
        return jsonify({
            "connected": True,
            "message": "TEST MODE: fake connection success"
        })

    if connected:
        return jsonify({
            "connected": True,
            "message": "already connected"
        })

    connected = bebop.connect(10)

    return jsonify({
        "connected": connected
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

    bebop.fly_direct(roll=0, pitch=10, yaw=0, vertical_movement=0, duration=1)

    return jsonify({"status": "forward success"})


@app.route("/backward", methods=["GET", "POST"])
def backward():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("backward")

    bebop.fly_direct(roll=0, pitch=-10, yaw=0, vertical_movement=0, duration=1)

    return jsonify({"status": "backward success"})


@app.route("/left", methods=["GET", "POST"])
def left():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("left")

    bebop.fly_direct(roll=-10, pitch=0, yaw=0, vertical_movement=0, duration=1)

    return jsonify({"status": "left success"})


@app.route("/right", methods=["GET", "POST"])
def right():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("right")

    bebop.fly_direct(roll=10, pitch=0, yaw=0, vertical_movement=0, duration=1)

    return jsonify({"status": "right success"})


@app.route("/up", methods=["GET", "POST"])
def up():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("up")

    bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=10, duration=1)

    return jsonify({"status": "up success"})


@app.route("/down", methods=["GET", "POST"])
def down():
    error = check_connection()
    if error:
        return error

    if TEST_MODE:
        return test_response("down")

    bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-10, duration=1)

    return jsonify({"status": "down success"})


@app.route("/disconnect", methods=["GET", "POST"])
def disconnect():
    global connected

    if TEST_MODE:
        connected = False
        return jsonify({
            "status": "TEST MODE disconnected"
        })

    if connected:
        bebop.disconnect()
        connected = False

    return jsonify({
        "status": "disconnected"
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )