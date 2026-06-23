import os
os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")

from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI


def user_code(args):
    print("Camera running...")


bebop = Bebop(drone_type="Bebop2")

success = bebop.connect(10)
print("Connection:", success)

if success:

    print("Starting camera")

    # Try to improve video stream stability
    try:
        if hasattr(bebop, "set_video_stream_mode"):
            bebop.set_video_stream_mode("high_reliability")
            print("Video mode set to high reliability")
        else:
            print("set_video_stream_mode not available in this pyparrot version")
    except Exception as e:
        print("Could not set video stream mode:", e)

    bebopVision = DroneVisionGUI(
        bebop,
        is_bebop=True,
        user_code_to_run=user_code,
        user_args=None,
        buffer_size=30
    )

    bebopVision.open_video()

else:
    print("Could not connect.")