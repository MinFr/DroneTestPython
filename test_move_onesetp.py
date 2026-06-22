from pyparrot.Bebop import Bebop

bebop = Bebop(drone_type="Bebop2")

print("Connecting to drone...")

success = bebop.connect(10)

print("Connection:", success)

if success:

    bebop.ask_for_state_update()
    bebop.smart_sleep(2)

    print("Battery:", bebop.sensors.battery)
    print("State:", bebop.sensors.flying_state)

    print("Taking off")
    bebop.safe_takeoff(10)

    bebop.smart_sleep(2)

    print("Moving forward")

    bebop.fly_direct(
        roll=0,  # gauche - /droite +               
        pitch=10,  # avancer + // reculer -             
        yaw=0,    # tourner              
        vertical_movement=0,  # monter + /descendre +  
        duration=15  # secondes       
    )

    bebop.smart_sleep(2)

    print("Landing")
    bebop.safe_land(10)

    bebop.smart_sleep(2)

    bebop.disconnect()

    print("Done.")

else:
    print("Could not connect.")