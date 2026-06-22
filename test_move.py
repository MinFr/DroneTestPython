 #   print("Move forward slowly...")
 #   bebop.fly_direct(
 #       roll=0, # gauche - /droite +
 #       pitch=10,# avancer + // reculer -
 #       yaw=0,# tourner
 #       vertical_movement=0,# monter + /descendre + 
 #      duration=1 # secondes
 #   )


from pyparrot.Bebop import Bebop

bebop = Bebop(drone_type="Bebop2")

success = bebop.connect(10)
print("Connection:", success)

if success:
    try:
        bebop.ask_for_state_update()
        bebop.smart_sleep(2)

        print("Battery:", bebop.sensors.battery)
        print("State:", bebop.sensors.flying_state)

        print("Taking off...")
        bebop.safe_takeoff(10)
        bebop.smart_sleep(3)

        print("Avancer")
        bebop.fly_direct(roll=0, pitch=10, yaw=0, vertical_movement=0, duration=1)
        bebop.smart_sleep(1)

        print("Aller à droite")
        bebop.fly_direct(roll=10, pitch=0, yaw=0, vertical_movement=0, duration=1)
        bebop.smart_sleep(1)

        print("Reculer")
        bebop.fly_direct(roll=0, pitch=-10, yaw=0, vertical_movement=0, duration=1)
        bebop.smart_sleep(1)

        print("Aller à gauche")
        bebop.fly_direct(roll=-10, pitch=0, yaw=0, vertical_movement=0, duration=1)
        bebop.smart_sleep(1)

        print("Atterrir...")
        bebop.safe_land(10)

    except Exception as e:
        print("Error:", e)
        print("urgence.")
        bebop.emergency_land()

    finally:
        bebop.disconnect()
        print("Disconnecte.")
else:
    print("Could not connect.")