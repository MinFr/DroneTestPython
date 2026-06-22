from pyparrot.Bebop import Bebop

bebop = Bebop(drone_type="Bebop2")

print("Connecting...")
success = bebop.connect(10)
print("Connection:", success)

if success:
    bebop.ask_for_state_update()
    bebop.smart_sleep(2)

    print("Battery:", bebop.sensors.battery)
    print("State before:", bebop.sensors.flying_state)

    print("Taking off...")
    bebop.safe_takeoff(10)

    bebop.smart_sleep(5)

    print("Landing...")
    bebop.safe_land(10)

    bebop.smart_sleep(2)

    print("State after:", bebop.sensors.flying_state)

    bebop.disconnect()
    print("Done.")
else:
    print("Could not connect.")