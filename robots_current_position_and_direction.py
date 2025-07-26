from controller import Robot, Camera, CameraRecognitionObject, InertialUnit, DistanceSensor
import math
import random
import numpy as np

# ── Initialization ────────────────────────────────────────────────────────────
robot = Robot()
timestep   = int(robot.getBasicTimeStep())
max_speed  = 10

# getting the position sensors
lps = robot.getDevice('left wheel sensor')
rps = robot.getDevice('right wheel sensor')
lps.enable(timestep)
rps.enable(timestep)


# Target GPS position (center of a specific grid cell)
goal_x = -0.875
goal_y = -0.875

# devices
lidar        = robot.getDevice('lidar');        lidar.enable(timestep); lidar.enablePointCloud()
gps          = robot.getDevice('gps');          gps.enable(timestep)
imu          = robot.getDevice('inertial unit'); imu.enable(timestep)
left_motor   = robot.getDevice('left wheel motor');  left_motor.setPosition(float('inf')); left_motor.setVelocity(0)
right_motor  = robot.getDevice('right wheel motor'); right_motor.setPosition(float('inf')); right_motor.setVelocity(0)


# Constants for mapping
CELL_SIZE = 0.25  # 8x8 grid for a 2x2 arena
ORIGIN_X = -1.0   # Bottom-left corner x
ORIGIN_Z = -1.0   # Bottom-left corner z
GRID_SIZE = 8     # 8x8 grid
WORLD_MIN = -1.0  # Minimum world coordinate (bottom-left corner)
WORLD_MAX = 1.0   # Maximum world coordinate (top-right corner)

# Direction indicators
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


# Map representation: 0=free, 1=wall, '?'=unknown, 'V'=visited
# Each cell has 5 elements: [top, right, bottom, left, visited]
maze = np.zeros((GRID_SIZE, GRID_SIZE, 5), dtype=int)
# Initialize the maze with walls
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        if y == GRID_SIZE - 1:
            maze[x][y][0] = 1  # Top wall (highest y)
        if x == GRID_SIZE - 1:
            maze[x][y][1] = 1  # Right wall (highest x)
        if y == 0:
            maze[x][y][2] = 1  # Bottom wall (lowest y)
        if x == 0:
            maze[x][y][3] = 1  # Left wall (lowest x)
    
# changed from LAB 3: VNESW to LAB 4: VWNES
# Replace your current print_maze() with:
def print_maze():
    print("________________________________________________________________________________")
    
    # Print matrix rows from 7 down to 0 (top to bottom)
    for matrix_row in range(GRID_SIZE-1, -1, -1):
        
        # Get visited status
        visited_chars = []
        for matrix_col in range(8):
            if maze[matrix_col][matrix_row][4] == 0:  # Note: [x][y] indexing
                visited_chars.append("?")
            else:
                visited_chars.append("V")
        
        # Print walls and visited status
        top_row = "|  " + "\t  ".join(str(maze[matrix_col][matrix_row][0]) for matrix_col in range(8)) + "    |"
        print(top_row)
        
        middle_parts = []
        for matrix_col in range(8):
            cell_part = f"{maze[matrix_col][matrix_row][3]} {visited_chars[matrix_col]} {maze[matrix_col][matrix_row][1]}"
            middle_parts.append(cell_part)
        middle_row = "|" + "\t".join(middle_parts) + "  |"
        print(middle_row)
        
        bottom_row = "|  " + "\t  ".join(str(maze[matrix_col][matrix_row][2]) for matrix_col in range(8)) + "    |"
        print(bottom_row)
        
        if matrix_row == 0:  # Changed from 7 to 0 since we're printing in reverse
            print("|_______________________________________________________________________________|\n")
        else:
            print("|                                                                               |")
def gps_to_cell(x, y):
    # raw indices 0..7
    gx = int((x - WORLD_MIN) // CELL_SIZE)
    gy = int((y - WORLD_MIN) // CELL_SIZE)
    
    # No flipping needed - this will make (0,0) the bottom-left
    
    # clamp into [0,7]
    gx = max(0, min(GRID_SIZE-1, gx))
    gy = max(0, min(GRID_SIZE-1, gy))
    return gx, gy


def get_robot_direction():
    """Get the robot's current direction based on IMU yaw."""
    yaw = imu.getRollPitchYaw()[2]  # Get yaw in radians
    if -math.pi/4 <= yaw < math.pi/4:
        return EAST
    elif math.pi/4 <= yaw < 3*math.pi/4:
        return NORTH
    elif -3*math.pi/4 <= yaw < -math.pi/4:
        return WEST
    else:
        return SOUTH

def is_front_clear():
    lidar_values = lidar.getRangeImage()
    # For e-puck, front is usually the middle index
    front_distance = lidar_values[len(lidar_values)//2]
    return front_distance > CELL_SIZE * 0.8  # 0.8 to be safe



def mark_current_cell_visited(x,y):
    maze[x][y][4] = 1  # Mark as visited




while robot.step(timestep) != -1:
    pos = gps.getValues()
    x, y, z = pos
    gx, gy = gps_to_cell(x, y)

    
    left_motor.setVelocity(max_speed)
    right_motor.setVelocity(max_speed)
    print("matrix position before:", maze[gx][gy])
    mark_current_cell_visited(gx, gy)
    print_maze()
    print(f"Current cell visited: ({gx}, {gy})")
    print("matrix position after:", maze[gx][gy])

    
