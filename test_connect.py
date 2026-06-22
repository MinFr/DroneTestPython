from pyparrot.Bebop import Bebop

bebop = Bebop(drone_type="Bebop2")

print("Connecting to Bebop 2...")
success = bebop.connect(10)

print("Connection result:", success)

if success:
    print("Connected.")
    bebop.ask_for_state_update()
    bebop.smart_sleep(2)

    print("Battery:", bebop.sensors.battery)
    print("Flying state:", bebop.sensors.flying_state)

    bebop.disconnect()
    print("Disconnected.")
else:
    print("Connection failed.")