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
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if i == 0:
            maze[i][j][0] = 1  # Top wall
        if j == GRID_SIZE - 1:
            maze[i][j][1] = 1  # Right wall
        if i == GRID_SIZE - 1:
            maze[i][j][2] = 1  # Bottom wall
        if j == 0:
            maze[i][j][3] = 1  # Left wall
    
def print_maze():
    """Print the maze with correct row/col indexing but simple separators."""
    
    # Fixed simple header instead of dynamic underscores
    print("=========================== MAZE ==========================")

    for row in range(GRID_SIZE):  # row = y-axis
        
        #  Top walls of this row
        top_row = "|  " + "\t  ".join(str(maze[row][col][0]) for col in range(GRID_SIZE)) + "    |"
        
        #  Middle row with left/right walls + visited info
        middle_row = "|" + "\t".join(
            f"{maze[row][col][3]} {'V' if maze[row][col][4] else '?'} {maze[row][col][1]}"
            for col in range(GRID_SIZE)
        ) + "  |"
        
        #  Bottom walls of this row
        bottom_row = "|  " + "\t  ".join(str(maze[row][col][2]) for col in range(GRID_SIZE)) + "    |"
        
        # Print the row contents
        print(top_row)
        print(middle_row)
        print(bottom_row)
        
        # Instead of complex underscores, just a simple fixed separator
        print("___________________________________________________________")
    
    # Footer separator
    print("==========================================================\n")

def gps_to_cell(x, y):
    """
    Map world coords X,Y in [–1,+1] to grid cols/rows [0..7],
    with (–1,–1) → (0,0) in the bottom‐left and (+1,+1) → (7,7) top‐right.
    """
    # raw indices 0..7
    gx = int((x - WORLD_MIN) // CELL_SIZE)
    gy = int((y - WORLD_MIN) // CELL_SIZE)
    # flip gy so that high Y → high row
    gy = (GRID_SIZE - 1) - gy

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

def move_forward_one_cell():
    start_pos = gps.getValues()
    while robot.step(timestep) != -1:
        if not is_front_clear():
            left_motor.setVelocity(0)
            right_motor.setVelocity(0)
            print("Obstacle ahead!")
            return False
        left_motor.setVelocity(max_speed)
        right_motor.setVelocity(max_speed)
        pos = gps.getValues()
        dx = pos[0] - start_pos[0]
        dz = pos[2] - start_pos[2]
        distance = math.sqrt(dx*dx + dz*dz)
        if distance >= CELL_SIZE * 0.95:
            break
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)
    return True

def mark_current_cell_visited(x,y):
    maze[x][y][4] = 1  # Mark as visited



while robot.step(timestep) != -1:
    pos = gps.getValues()
    x, y, z = pos
    gx, gy = gps_to_cell(x, y)
    print(f"GPS: ({x:.3f}, {y:.3f}, {z:.3f}) -> Grid Cell: ({gx}, {gy})")
    # print("robot pointing", get_robot_direction())
    # move_forward_one_cell()
    mark_current_cell_visited(gx, gy)
    # maze[7][1][4] = 1 
    print_maze()
    print(f"Current cell visited: ({gx}, {gy})")

    
