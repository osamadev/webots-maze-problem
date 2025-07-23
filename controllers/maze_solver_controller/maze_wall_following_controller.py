# maze_wall_following_controller.py
from controller import Robot, DistanceSensor

# create robot instance
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# max wheel speed
max_speed = 6.28

# initialize motors
left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# enable the 8 IR sensors
prox_sensors = []
for ind in range(8):
    sensor_name = 'ps' + str(ind)         
    prox_sensors.append(robot.getDevice(sensor_name))
    prox_sensors[ind].enable(timestep)

# main loop
while robot.step(timestep) != -1:

    # read all sensors
    for ind in range(8):
        print('ind: {}, val: {}'.format(ind,
                                        prox_sensors[ind].getValue()))

    # simple wall‑following logic
    left_wall  = prox_sensors[5].getValue() > 80   # right‑hand wall follower
    front_wall = prox_sensors[7].getValue() > 80

    left_speed  = max_speed
    right_speed = max_speed

    if front_wall:
        print("Turn right")
        left_speed  =  max_speed
        right_speed = -max_speed

    else:
        if left_wall:
            print("Drive forward")
            left_speed  = max_speed
            right_speed = max_speed
        else:
            print("Turn left")
            left_speed  = -max_speed / 8
            right_speed =  max_speed

    # send speeds to wheels
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)