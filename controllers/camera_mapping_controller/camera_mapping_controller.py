from controller import Robot
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from collections import deque
import json

# Set simulation time step and robot speed constants
TIME_STEP = 64  # Simulation time step in ms
MAX_SPEED = 6.28

SAFE_DISTANCE = 0.3  # Minimum safe distance from obstacle
TURN_SPEED_MOD = 0.8
FORWARD_TIME = 61  # Time steps for forward movement over 1 grid

GRID_SIZE = 0.5  # Size of each grid cell in meters
GRID_X = 5
GRID_Y = 5

# Cardinal directions and mappings
DIR_N, DIR_E, DIR_S, DIR_W = 0, 1, 2, 3

# Turning parameters
KP_TURN = 1.2
MIN_TURN_FRAC = 0.18
MAX_TURN_FRAC = 0.7
YAW_TOL = np.deg2rad(2.0)
TURN_HOLD_STEPS = 3

REF_DISTANCES = [0.5, 0.4, 0.3, 0.2, 0.1]  # For distance estimation from wall images
IMG_MATCH_THRESHOLD = 0.7  # SSIM threshold for wall detection

# Initialize Webots robot and device handles
robot = Robot()
cameras = [
    robot.getDevice('cameraFront'),
    robot.getDevice('cameraLeft'),
    robot.getDevice('cameraBack'),
    robot.getDevice('cameraRight'),
]
for camera in cameras:
    camera.enable(TIME_STEP)
    width = camera.getWidth()
    height = camera.getHeight()

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

imu = robot.getDevice('inertial unit')
imu.enable(TIME_STEP)


def angle_wrap(a):
    """
    Wraps angle 'a' into the range [0, 2*pi).
    """
    return (a + 2 * np.pi) % (2 * np.pi)

def angle_diff(current, target):
    """
    Computes shortest signed difference between two angles (radians).
    Result is in range [-pi, pi].
    """
    d = (target - current + np.pi) % (2 * np.pi) - np.pi
    return d

def preprocess(img, bottom_crop_frac=0.0):
    """
    Converts BGR image to normalized grayscale.
    Optionally crops the bottom fraction (not used).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype("float32") / 255

# Load and preprocess reference wall images for visual wall recognition
ref_imgs = []
for i in range(1, 6):
    ref_bgr = cv2.imread(f"./wall_image{i}.png")
    if ref_bgr is None:
        raise FileNotFoundError(f"Cannot read reference image at: ./wall_image{i}.png")
    ref_bgr = preprocess(ref_bgr)
    ref_imgs.append(ref_bgr)

def calc_image_similarity(img, ref):
    """
    Returns the max SSIM between img (and its mirror) and a reference image.
    Useful for wall recognition with possible orientation changes.
    """
    score1 = ssim(img, ref, data_range=1)
    score2 = ssim(np.flip(img, axis=1), ref, data_range=1)
    return max(score1, score2)

def recognise_walls():
    """
    Recognizes walls in all four directions using the cameras and reference images.
    Returns a list of bools for (front, left, back, right).
    """
    is_wall = [False] * 4
    for i, camera in enumerate(cameras):
        image_data = camera.getImage()
        img = np.frombuffer(image_data, np.uint8).reshape((height, width, -1))
        img = img[..., :3].copy()
        img = preprocess(img)
        match_score = max([calc_image_similarity(img, ref_bgr) for ref_bgr in ref_imgs])
        if match_score > IMG_MATCH_THRESHOLD:
            is_wall[i] = True
    return is_wall

def estimate_front_distance():
    """
    Estimates distance to the wall in front using SSIM matching with reference images.
    Returns an approximate distance or 10 if nothing matches.
    """
    camera = cameras[0]  # Front camera
    image_data = camera.getImage()
    img = np.frombuffer(image_data, np.uint8).reshape((height, width, -1))
    img = img[..., :3].copy()
    img = preprocess(img)
    match_score, max_idx = max([(calc_image_similarity(img, ref_bgr), i) for i, ref_bgr in enumerate(ref_imgs)])
    if match_score < IMG_MATCH_THRESHOLD:
        return 10  # Too far, not detected
    return REF_DISTANCES[max_idx]

def print_maze(maze):
    """
    Prints the maze state for debugging, showing visited status and wall encoding.
    """
    print("___________________________________")
    rows = len(maze)
    cols = len(maze[0])
    for y in range(rows - 1, -1, -1):
        visited_chars = []
        for x in range(cols):
            cell = maze[y][x]
            visited_chars.append("V" if '?' not in cell else "?")
        top_row = "|  " + "\t  ".join(str(maze[y][x][0]) for x in range(cols)) + "   |"
        print(top_row)
        middle_parts = []
        for x in range(cols):
            cell = maze[y][x]
            cell_part = f"{cell[3]} {visited_chars[x]} {cell[1]}"
            middle_parts.append(cell_part)
        middle_row = "|" + "\t".join(middle_parts) + " |"
        print(middle_row)
        bottom_row = "|  " + "\t  ".join(str(maze[y][x][2]) for x in range(cols)) + "   |"
        print(bottom_row)
        print("|__________________________________\n" if y == 0 else "|                           ")

def format_radians(theta):
    """
    Normalizes an angle to [0, 2*pi).
    """
    if 0 <= theta < 2 * np.pi:
        return theta
    if theta < 0:
        return format_radians(theta + 2 * np.pi)
    return format_radians(theta - 2 * np.pi)

def imu_yaw():
    """
    Reads and normalizes the robot's yaw angle from the IMU.
    """
    _, _, y = imu.getRollPitchYaw()
    return format_radians(y)

def get_facing(state_var):
    """
    Determines which direction (N/E/S/W) the robot is facing, given its yaw.
    """
    theta = format_radians(state_var[2])
    idx = int((theta + np.pi / 4.0) // (np.pi / 2.0)) % 4
    mapping = {0: DIR_E, 1: DIR_N, 2: DIR_W, 3: DIR_S}
    return mapping[idx]

def write_wall(x, y, side, has_wall):
    """
    Writes a wall (1) or no-wall (0) status into the maze for a given cell and side.
    """
    if 0 <= x < GRID_X and 0 <= y < GRID_Y:
        if state_dict['maze'][y][x][side] == '?':
            state_dict['maze'][y][x][side] = 1 if has_wall else 0

def harmonise_map():
    """
    Ensures wall information is consistent between neighboring cells (shared walls).
    """
    for x in range(GRID_X):
        for y in range(GRID_Y - 1):
            if state_dict['maze'][y][x][0] == '?' and state_dict['maze'][y + 1][x][2] != '?':
                state_dict['maze'][y][x][0] = state_dict['maze'][y + 1][x][2]
            if state_dict['maze'][y + 1][x][2] == '?' and state_dict['maze'][y][x][0] != '?':
                state_dict['maze'][y + 1][x][2] = state_dict['maze'][y][x][0]
    for x in range(GRID_X - 1):
        for y in range(GRID_Y):
            if state_dict['maze'][y][x][1] == '?' and state_dict['maze'][y][x + 1][3] != '?':
                state_dict['maze'][y][x][1] = state_dict['maze'][y][x + 1][3]
            if state_dict['maze'][y][x + 1][3] == '?' and state_dict['maze'][y][x][1] != '?':
                state_dict['maze'][y][x + 1][3] = state_dict['maze'][y][x][1]

def scan():
    """
    Updates the map at the robot's current location by detecting surrounding walls.
    """
    is_wall = recognise_walls()
    facing = get_facing(state_dict['state_var'])
    x = int(state_dict['state_var'][0])
    y = int(state_dict['state_var'][1])
    rel_front = facing
    rel_left = (facing + 3) % 4
    rel_back = (facing + 2) % 4
    rel_right = (facing + 1) % 4
    write_wall(x, y, rel_front, is_wall[0])
    write_wall(x, y, rel_left, is_wall[1])
    write_wall(x, y, rel_back, is_wall[2])
    write_wall(x, y, rel_right, is_wall[3])
    harmonise_map()

def _turn_controller(delta_target):
    """
    Turns the robot by a specified angle using the IMU for closed-loop feedback.
    Handles left/right turns by updating the goal_stack.
    """
    if not state_dict.get('turn_flag', False):
        state_dict['turn_flag'] = True
        cur_yaw = imu_yaw()
        state_dict['turn_target_yaw'] = angle_wrap(cur_yaw + delta_target)
        state_dict['turn_hold'] = TURN_HOLD_STEPS

    cur_yaw = imu_yaw()
    err = angle_diff(cur_yaw, state_dict['turn_target_yaw'])

    if abs(err) > YAW_TOL:
        speed_frac = min(MAX_TURN_FRAC, max(MIN_TURN_FRAC, abs(KP_TURN * err)))
        if err > 0:
            left_motor.setVelocity(-speed_frac * MAX_SPEED * TURN_SPEED_MOD)
            right_motor.setVelocity(+speed_frac * MAX_SPEED * TURN_SPEED_MOD)
        else:
            left_motor.setVelocity(+speed_frac * MAX_SPEED * TURN_SPEED_MOD)
            right_motor.setVelocity(-speed_frac * MAX_SPEED * TURN_SPEED_MOD)
        state_dict['turn_hold'] = TURN_HOLD_STEPS
    else:
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        state_dict['turn_hold'] -= 1
        if state_dict['turn_hold'] <= 0:
            state_dict['turn_flag'] = False
            state_dict['goal_stack'].pop(-1)
            state_dict['goal_stack'].append(('calibrate_state', None))
            return

def left_turn(theta=None):
    """
    Initiates a left (counter-clockwise) turn by 90 degrees or a specified angle.
    """
    _turn_controller(theta if theta is not None else (np.pi / 2))

def right_turn(theta=None):
    """
    Initiates a right (clockwise) turn by 90 degrees or a specified angle.
    """
    _turn_controller(-(theta if theta is not None else (np.pi / 2)))

def forward(dist=None):
    """
    Moves the robot forward (or backward) a specified distance (defaults to one grid cell).
    Uses the IMU to update internal state after moving.
    """
    if dist is None:
        dist = GRID_SIZE
    if not state_dict['forward_flag']:
        state_dict['forward_flag'] = True
        if dist >= 0:
            state_dict['forward_counter'] = int(FORWARD_TIME * dist / GRID_SIZE)
            state_dict['reverse_indicator'] = 1
        else:
            state_dict['forward_counter'] = int(FORWARD_TIME * abs(dist) / GRID_SIZE)
            state_dict['reverse_indicator'] = -1

    fdist = estimate_front_distance()
    if fdist < SAFE_DISTANCE:
        v = 0.4 * MAX_SPEED * state_dict['reverse_indicator']
    else:
        v = 1.0 * MAX_SPEED * state_dict['reverse_indicator']

    current_facing = get_facing(state_dict['state_var'])
    if state_dict['forward_counter'] > 0:
        left_motor.setVelocity(MAX_SPEED * state_dict['reverse_indicator'])
        right_motor.setVelocity(MAX_SPEED * state_dict['reverse_indicator'])
        state_dict['forward_counter'] -= 1
    else:
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        state_dict['forward_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('calibrate_state', None))
        step = dist / GRID_SIZE
        # Update robot state according to the direction moved
        if current_facing == DIR_N:
            state_dict['state_var'][1] += step
        elif current_facing == DIR_E:
            state_dict['state_var'][0] += step
        elif current_facing == DIR_S:
            state_dict['state_var'][1] -= step
        elif current_facing == DIR_W:
            state_dict['state_var'][0] -= step

def calibrate_state():
    """
    Ensures robot's heading is aligned with the closest cardinal direction.
    Triggers correction turns if needed.
    """
    state_dict['state_var'][2] = imu_yaw()
    cur_yaw = angle_wrap(state_dict['state_var'][2])
    cur_dir = get_facing(state_dict['state_var'])
    target_yaw_map = {
        DIR_E: 0.0,
        DIR_N: np.pi / 2.0,
        DIR_W: np.pi,
        DIR_S: 3.0 * np.pi / 2.0
    }
    target_yaw = angle_wrap(target_yaw_map[cur_dir])
    delta = angle_diff(cur_yaw, target_yaw)
    if abs(delta) > YAW_TOL:
        if state_dict['goal_stack'] and state_dict['goal_stack'][-1][0] == 'calibrate_state':
            state_dict['goal_stack'].pop(-1)
        if delta > 0:
            state_dict['goal_stack'].append(('left_turn', delta))
        else:
            state_dict['goal_stack'].append(('right_turn', abs(delta)))
        return
    scan()
    if state_dict['goal_stack'] and state_dict['goal_stack'][-1][0] == 'calibrate_state':
        state_dict['goal_stack'].pop(-1)

def side_open(maze, x, y, side):
    """
    Returns True if the given side of the cell is open (not a wall).
    """
    v = maze[y][x][side]
    return v != 1

def neighbors(maze, x, y):
    """
    Yields coordinates of neighboring cells that are accessible (no wall in between).
    """
    if y + 1 < GRID_Y and side_open(maze, x, y, DIR_N):
        yield (x, y + 1)
    if x + 1 < GRID_X and side_open(maze, x, y, DIR_E):
        yield (x + 1, y)
    if y - 1 >= 0 and side_open(maze, x, y, DIR_S):
        yield (x, y - 1)
    if x - 1 >= 0 and side_open(maze, x, y, DIR_W):
        yield (x - 1, y)

def bfs_next_step(maze, start, goal):
    """
    Finds the next cell towards goal from start using BFS. Returns None if unreachable.
    """
    if start == goal:
        return start
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nb in neighbors(maze, cur[0], cur[1]):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    if goal not in parent:
        return None
    node = goal
    while parent[node] is not None and parent[node] != start:
        node = parent[node]
    return node

def is_frontier_cell(x, y):
    """
    Returns True if the cell at (x, y) has any unknown walls ('?'), i.e., is a mapping frontier.
    """
    return '?' in state_dict['maze'][y][x]

def nearest_frontier_from(start_xy):
    """
    Finds the closest frontier cell (cell with unknown walls) from the given position.
    Uses BFS search.
    """
    sx, sy = start_xy
    if is_frontier_cell(sx, sy):
        return (sx, sy)
    q = deque([start_xy])
    seen = {start_xy}
    while q:
        cx, cy = q.popleft()
        for nx, ny in neighbors(state_dict['maze'], cx, cy):
            if (nx, ny) not in seen:
                if is_frontier_cell(nx, ny):
                    return (nx, ny)
                seen.add((nx, ny))
                q.append((nx, ny))
    return None

def map_current():
    """
    Scans and updates the current cell. Pops the action if already fully mapped.
    """
    cx = int(state_dict['state_var'][0])
    cy = int(state_dict['state_var'][1])
    scan()
    if '?' not in state_dict['maze'][cy][cx]:
        state_dict['goal_stack'].pop(-1)
    else:
        state_dict['goal_stack'].append(('right_turn', None))

def explore_step():
    """
    Main exploration logic: plans route to nearest unexplored cell, or finishes if done.
    """
    cx = int(state_dict['state_var'][0])
    cy = int(state_dict['state_var'][1])
    if is_frontier_cell(cx, cy):
        state_dict['goal_stack'].append(('map_current', None))
        return
    # print_maze(state_dict["maze"])
    target = nearest_frontier_from((cx, cy))
    if target is None:
        with open('map.json', 'w') as f:
            json.dump(state_dict['maze'], f)
        print("Mapping complete.")
        state_dict['goal_stack'].pop(-1)
        return
    if (cx, cy) == target:
        print("Got target")
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('map_current', None))
    else:
        state_dict['goal_stack'].append(('route', target))

def route(target_xy):
    """
    Plans a move towards the given target cell using BFS. Adds move_next to the stack.
    """
    cx = int(state_dict['state_var'][0])
    cy = int(state_dict['state_var'][1])
    cur = (cx, cy)
    goal = (int(target_xy[0]), int(target_xy[1]))
    if cur == goal:
        state_dict['goal_stack'].pop(-1)
        return
    m = state_dict['maze']
    next_cell = bfs_next_step(m, cur, goal)
    if next_cell is None:
        d_now = abs(goal[0] - cx) + abs(goal[1] - cy)
        cands = [((nx, ny), abs(goal[0] - nx) + abs(goal[1] - ny)) for (nx, ny) in neighbors(m, cx, cy)]
        cands.sort(key=lambda t: t[1])
        if not cands or cands[0][1] >= d_now:
            state_dict['goal_stack'].pop(-1)
            return
        next_cell = cands[0][0]
    state_dict['goal_stack'].append(("move_next", (next_cell[0], next_cell[1])))

def move_next(next_xy):
    """
    Orients the robot towards the next cell and plans necessary turns/forward movements.
    """
    cx = int(state_dict['state_var'][0])
    cy = int(state_dict['state_var'][1])
    dx = next_xy[0] - cx
    dy = next_xy[1] - cy
    if abs(dx) + abs(dy) != 1:
        state_dict['goal_stack'].pop(-1)
        return
    if dy == 1:
        desired_dir = DIR_N
    elif dy == -1:
        desired_dir = DIR_S
    elif dx == 1:
        desired_dir = DIR_E
    else:
        desired_dir = DIR_W
    cur_dir = get_facing(state_dict['state_var'])
    delta = (desired_dir - cur_dir) % 4
    state_dict['goal_stack'].pop(-1)
    state_dict['goal_stack'].append(('forward', GRID_SIZE))
    if delta == 1:
        state_dict['goal_stack'].append(('right_turn', None))
    elif delta == 3:
        state_dict['goal_stack'].append(('left_turn', None))
    elif delta == 2:
        state_dict['goal_stack'].append(('left_turn', None))
        state_dict['goal_stack'].append(('left_turn', None))

def race(target_xy):
    """
    Follows a pre-mapped path to the target cell. Used in Path Following Stage.
    """
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    if current_x == target_xy[0] and current_y == target_xy[1]:
        state_dict['goal_stack'].pop(-1)
    else:
        state_dict['goal_stack'].append(("route", (target_xy[0], target_xy[1])))

def check_complete_mapping():
    """
    Checks if the mapping phase is done (no unexplored/frontier cells).
    """
    maze = state_dict["maze"]
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if is_frontier_cell(j, i):
                return False
    return True

# Map behaviour names to functions for stack execution
behaviours = {
    "explore_step": explore_step,
    "map_current": map_current,
    "right_turn": right_turn,
    "left_turn": left_turn,
    "calibrate_state": calibrate_state,
    "route": route,
    "race": race,
    "move_next": move_next,
    "forward": forward
}

# Load map from file if available, else initialize blank map
try:
    with open('map.json') as f:
        maze = json.load(f)
except Exception:
    maze = [[['?', '?', '?', '?'] for _ in range(GRID_X)] for _ in range(GRID_Y)]

# Initialize robot state dictionary and goal stack
state_dict = {}
state_dict['maze'] = maze
state_dict['goal_stack'] = []
state_dict['turn_flag'] = False
state_dict['forward_flag'] = False
state_dict['move_flag'] = False
state_dict['state_var'] = [0.5, 0.5, imu_yaw()]

# Decide stage: path following if map complete, otherwise start mapping
if check_complete_mapping():
    print("Path Following Stage")
    state_dict['goal_stack'] = [("race", (1, 4)), ("calibrate_state", None)]
else:
    print("Mapping Stage")
    state_dict['goal_stack'] = [("race", (1, 4)), ("explore_step", None), ("calibrate_state", None)]

goal_stack_string_buffer = "->".join([str(goal) for goal in state_dict['goal_stack']])

# Main control loop: run as long as simulation is active
while robot.step(TIME_STEP) != -1:
    goal_stack_string = "->".join([str(goal) for goal in state_dict['goal_stack']])
    if goal_stack_string != goal_stack_string_buffer:
        print_maze(state_dict['maze'])
        print(goal_stack_string)
        goal_stack_string_buffer = goal_stack_string
    if len(state_dict['goal_stack']) > 0:
        plan = state_dict['goal_stack'][-1]
        if plan[1] is not None:
            behaviours[plan[0]](plan[1])
        else:
            behaviours[plan[0]]()
