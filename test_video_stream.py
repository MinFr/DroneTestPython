from pyparrot.Bebop import Bebop

bebop = Bebop(drone_type="Bebop2")

success = bebop.connect(10)
print("Connection:", success)

if success:
    print("Start video stream...")
    bebop.start_video_stream()
    bebop.smart_sleep(10)

    print("Stop video stream...")
    bebop.stop_video_stream()

    bebop.disconnect()
    print("Done.")
else:
    print("Could not connect.")