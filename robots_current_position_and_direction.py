from controller import Robot, Camera, CameraRecognitionObject, InertialUnit, DistanceSensor
import math
import random
import numpy as np

robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
max_speed= 10  # Set the maximum speed for the robot


# Add:
lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

# getting the position sensors
lps = robot.getDevice('left wheel sensor')
rps = robot.getDevice('right wheel sensor')
lps.enable(timestep)
rps.enable(timestep)

#enable imu
imu = robot.getDevice('inertial unit')
imu.enable(timestep)

# GPS
gps = robot.getDevice('gps')
gps.enable(timestep)

# Target GPS position (center of a specific grid cell)
goal_x = -0.875
goal_y = -0.875

# get handler to motors and set target position to infinity
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0)
rightMotor.setVelocity(0)

# Constants for mapping
CELL_SIZE = 0.25  # 8x8 grid for a 2x2 arena
ORIGIN_X = -1.0   # Bottom-left corner x
ORIGIN_Z = -1.0   # Bottom-left corner z
GRID_SIZE = 8     # 8x8 grid
WORLD_MIN = -1.0  # Minimum world coordinate (bottom-left corner)
WORLD_MAX = 1.0   # Maximum world coordinate (top-right corner)



def gps_to_cell(x, y):
    gx = int((x - WORLD_MIN) // CELL_SIZE)   # horizontal
    gy = int((y - WORLD_MIN) // CELL_SIZE)   # vertical (no inversion!)
    gx = max(0, min(GRID_SIZE - 1, gx))
    gy = max(0, min(GRID_SIZE - 1, gy))
    return gx, gy

def get_robot_direction():
    """Get the robot's current direction based on IMU yaw."""
    yaw = imu.getRollPitchYaw()[2]  # Get yaw in radians
    if -math.pi/4 <= yaw < math.pi/4:
        return "EAST"
    elif math.pi/4 <= yaw < 3*math.pi/4:
        return "NORTH"
    elif -3*math.pi/4 <= yaw < -math.pi/4:
        return "WEST"
    else:
        return "SOUTH"

while robot.step(timestep) != -1:
    pos = gps.getValues()
    x, y, z = pos  # Webots GPS returns (x, altitude, y)
    
    gx, gy = gps_to_cell(x, y)
    print(f"GPS: ({x:.3f}, {y:.3f}, {z:.3f}) -> Grid Cell: ({gx}, {gy})")
    print("robot pointing", get_robot_direction())
    
