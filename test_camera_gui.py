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

    bebopVision = DroneVisionGUI(
        bebop,
        is_bebop=True,
        user_code_to_run=user_code,
        user_args=None, 
        buffer_size=200 # Buffer de 200 images
    )

    bebopVision.open_video()

else:
    print("Could not connect.")