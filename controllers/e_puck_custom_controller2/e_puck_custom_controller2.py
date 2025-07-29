from controller import Robot, Display
import cv2
import math
import numpy as np
from collections import defaultdict
import json

TIME_STEP = 64
MAX_SPEED = 6.28
SAFE_DISTANCE = 0.3  # distance to trigger avoidance
TURN_TIME = 14       # steps to force a turn
TURN_SPEED_MOD = 0.8
FORWARD_TIME = 61
error = 0.07

robot = Robot()

# Motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Lidar
lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)

# create map
grid_x = 4
grid_y = 4
grid_size = 0.5

# Camera
camera = robot.getDevice('camera1')
camera.enable(TIME_STEP)
width = camera.getWidth()
height = camera.getHeight()
display = robot.getDevice("my_display") # Replace "my_display" with your display's name


# should try to read from file first
try:
    with open('map.json') as f:
        map = json.load(f)

except:
    map = [[['?','?','?','?'] for x in range(grid_x)] for y in range(grid_y)]


def wrap_angle(a):
    # map to (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

class ThetaKF:
    """1D KF for heading error (theta_adjustment). State: x ~ heading error."""
    def __init__(self, q=1e-4, init_var=1e-2):
        self.x = 0.0           # estimated heading error (rad)
        self.P = float(init_var)
        self.q = float(q)      # process noise power (rad^2 / s)

    def predict(self, dt, gyro_bias_comp=None):
        # Random-walk model for error; optionally nudge with a (small) bias if you have one
        # x_k|k-1 = x_{k-1}  (no control on error); P_k|k-1 = P + q*dt
        self.P += self.q * max(dt, 1e-3)

    def update(self, z, R):
        # z is an angle error in radians (measurement of same quantity)
        if z is None or R is None:
            return
        # Innovation
        y = wrap_angle(z - self.x)
        S = self.P + R
        K = self.P / S
        # Update
        self.x = wrap_angle(self.x + K * y)
        self.P = (1.0 - K) * self.P



# holding state variables in dict for easier management of variable scope
state_dict = {}
state_dict['theta_kf'] = ThetaKF(q=2e-4, init_var=1e-2)  # tune q/init_var as needed
state_dict['state_var'] = [0.5,0.5,0.0]
state_dict['goal_stack'] = [("race",(2,0)),("mapping",None),("calibrate_state",None)]
state_dict['map'] = map
state_dict['turn_flag'] = False
state_dict['forward_flag'] = False
state_dict['move_flag'] = False


def format_radians(theta):

    """Recursive function to keep radian values within range 0 - 2pi
       Assists with identifying 'get_facing' function."""

    # already in range - return existing value
    if theta >=0 and theta < 2 * np.pi:
        return theta
    # less than zero - recursively add 2pi until in range
    if theta < 0:
        theta = theta + 2 * np.pi
        return format_radians(theta)
    # greater than 2pi, recursively subtract 2pi until in range
    else:
        theta = theta - 2 * np.pi
        return format_radians(theta)

def get_facing(state_var):
    """ Return N,E,S,W (0,1,2,3) based on theta nearest """
    theta = format_radians(state_var[2])
    if theta > 7*2*np.pi/8 or theta <= 1*2*np.pi/8:
        return 0
    elif theta > 1*2*np.pi/8 and theta <= 3*2*np.pi/8:
        return 1
    elif theta > 3*2*np.pi/8 and theta <= 5*2*np.pi/8:
        return 2
    else:
    # better to put exception here
        return 3

def get_x_robot_ref(state_var):
    """Detemine x coord in robot orientated ref frame.
    Use x if facing N or S, y if E or W."""
    facing = get_facing(state_var)

    if facing == 0 or facing == 2:
        return state_var[0]
    else:
        return state_var[1]

def get_y_robot_ref(state_var):
    """Detemine y coord in robot orientated ref frame.
    Use y if facing N or S, x if E or W."""
    facing = get_facing(state_var)

    if facing == 0 or facing == 2:
        return state_var[1]
    else:
        return state_var[0]

def get_rel_wall_position_left(facing):
    """Multiplier to help manage relative wall position for different facings.
       Based on 'lefthand' wall, if facing North or West, the wall will be in a negative
       relative position to the x coordinate of the robot.  If East or South, it will be positive.
       Same function is used for righthand walls by switching the sign where relevant.
       """
    if facing == 0 or facing == 3:
        return -1
    else:
        return 1

def get_rel_wall_position_forward(facing):
    """Multiplier to help manage relative wall position for different facings.
       Based on 'forward wall', if facing North or East, the wall will be in a positive
       relative position to the y coordinate of the robot.  If South or West, it will be negative.
       """
    if facing == 0 or facing == 1:
        return 1
    else:
        return -1


def get_lidar_a(ranges,state_var):

    facing = get_facing(state_var)
    x_robot_ref = get_x_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_left(facing)

    start_alpha = (np.pi-2.9)/2
    sample = ranges[0:53:10]
    # alpha is angle relative to -x direction (robot frame)
    alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]

    sample_x = [x_robot_ref*grid_size + facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    pred_x = [(x_robot_ref+facing_multiplier*0.5)*grid_size for i in range(len(sample))]

    x_squared = [(pred_x[i]-sample_x[i])**2 for i in range(len(sample))]
    if sum(x_squared) < len(sample)*error**2:
        return True
    else:
        return False

def get_lidar_b(ranges,state_var):

    facing = get_facing(state_var)
    x_robot_ref = get_x_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_left(facing)

    start_alpha = (np.pi-2.9)/2 + 0.7
    sample = ranges[124:159:10]
    # alpha is angle relative to -x direction (robot frame)
    alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]

    sample_x = [x_robot_ref*grid_size + facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    pred_x = [(x_robot_ref+facing_multiplier*0.5)*grid_size for i in range(len(sample))]

    x_squared = [(pred_x[i]-sample_x[i])**2 for i in range(len(sample))]
    if sum(x_squared) < len(sample)*error**2:
        return True
    else:
        return False

def get_lidar_c(ranges,state_var):

    facing = get_facing(state_var)

    # y coord if N/S, x coord if E/W
    y_robot_ref = get_y_robot_ref(state_var)

    facing_multiplier = get_rel_wall_position_forward(facing)

    start_alpha = (np.pi-2.9)/2 + 1.3
    sample = ranges[230:282:10]
    # alpha is angle relative to forward direction
    alpha = [3.14/2-(start_alpha+i*10/512*2.9) for i in range(len(sample))]

    sample_y = [y_robot_ref*grid_size+facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    pred_y = [(y_robot_ref+facing_multiplier*0.5)*grid_size for i in range(len(sample))]


    y_squared = [(pred_y[i]-sample_y[i])**2 for i in range(len(sample))]
    if sum(y_squared) < len(sample)*error**2:
        return True
    else:
        return False



def get_lidar_d(ranges,state_var):

    facing = get_facing(state_var)
    x_robot_ref = get_x_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_left(facing)

    start_alpha = (np.pi-2.9)/2 + 2
    sample = ranges[353:388:10]
    # alpha is angle relative to -x direction (robot frame)
    alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]

    sample_x = [x_robot_ref*grid_size + facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    pred_x = [(x_robot_ref-facing_multiplier*0.5)*grid_size for i in range(len(sample))]

    x_squared = [(pred_x[i]-sample_x[i])**2 for i in range(len(sample))]
    if sum(x_squared) < len(sample)*error**2:
        return True
    else:
        return False


def get_lidar_e(ranges,state_var):

    facing = get_facing(state_var)
    x_robot_ref = get_x_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_left(facing)

    start_alpha = (np.pi-2.9)/2 + 2.6
    sample = ranges[459:512:10]
    # alpha is angle relative to -x direction (robot frame)
    alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]

    sample_x = [x_robot_ref*grid_size + facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    pred_x = [(x_robot_ref-facing_multiplier*0.5)*grid_size for i in range(len(sample))]

    x_squared = [(pred_x[i]-sample_x[i])**2 for i in range(len(sample))]
    if sum(x_squared) < len(sample)*error**2:
        return True
    else:
        return False

def check_closest_unmapped_square(x_y,map,unmapped):
    """Use wavefront algorithm to find closest unmapped squares.
       This function should only be called if there are unmapped squares."""
    map_distance = [['?' for x in range(grid_x)] for y in range(grid_y)]


    wavefront = 1
    wavefront_squares = defaultdict(set)

    wavefront_squares[1].add(x_y)



    while True:
        # print("could be this in check_closest_unmapped?")
        # if wavefront has found an unmapped square, just return it as the target
        for square in wavefront_squares[wavefront]:
            for unmapped_sq in unmapped:
                if square[0] == unmapped_sq[0] and square[1] == unmapped_sq[1]:
                    # print("found unmapped square at ",square[0],square[1])
                    return (square[0],square[1])

        for y in range(grid_y):
            for x in range(grid_x):
                for sq in wavefront_squares[wavefront]:
                    # if square is one east of one in last wavefront AND no wall on east
                    if x == sq[0] + 1 and y == sq[1] and map[sq[1]][sq[0]][1] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one north of one in last wavefront AND no wall on north
                    if x == sq[0] and y == sq[1] + 1  and map[sq[1]][sq[0]][0] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one west of one in last wavefront AND no wall on west
                    if x == sq[0] -1 and y == sq[1]  and map[sq[1]][sq[0]][3] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one south of one in last wavefront AND no wall on south
                    if x == sq[0] and y == sq[1] -1  and map[sq[1]][sq[0]][2] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
        wavefront += 1



def check_map(map,state_var):
    """check if unexplored squares.
       Return False if none, otherwise return closest square."""
    unmapped = []
    for y in range(grid_y):
        for x in range(grid_x):
            if '?' in map[y][x]:
                unmapped.append((x,y))
    if len(unmapped) == 0:

        return False
    else:
        current_x = int(state_var[0])
        current_y = int(state_var[1])

        return check_closest_unmapped_square((current_x,current_y),map,unmapped)






# define highest level plans

def mapping():


    nearest_unmapped = check_map(state_dict['map'],state_dict['state_var'])

    if not nearest_unmapped:

        state_dict['goal_stack'].pop(-1)
        with open('map.json','w') as f:
            json.dump(state_dict['map'],f)

    else:
        current_x = int(state_dict['state_var'][0])
        current_y = int(state_dict['state_var'][1])

        # if current square not fully mapped, add goal to map current square to goal stack
        if nearest_unmapped[0] == current_x and nearest_unmapped[1] == current_y:
            state_dict['goal_stack'].append(("map_current",None))
        else:
            state_dict['goal_stack'].append(("route",nearest_unmapped))



def map_current():


    scan()

    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])

    if '?' not in state_dict['map'][current_y][current_x]:
        state_dict['goal_stack'].pop(-1)

    else:
        state_dict['goal_stack'].append(('right_turn',None))



def scan():
    scan_pos = get_scan_positions()
    # print("scan_pos",scan_pos)
    ranges = lidar.getRangeImage()

    state_var = state_dict['state_var']
    # print("state_var",state_var)
    x_a = scan_pos[0][0]
    y_a = scan_pos[0][1]
    facing_a = scan_pos[0][2]
    if state_dict['map'][y_a][x_a][facing_a] == '?':
        if get_lidar_a(ranges,state_var):
            state_dict['map'][y_a][x_a][facing_a] = 1
        else:
            state_dict['map'][y_a][x_a][facing_a] = 0

    x_b = scan_pos[1][0]
    y_b = scan_pos[1][1]
    facing_b = scan_pos[1][2]
    # b can look off map. Check is in map
    if x_b>=0 and x_b<grid_x and y_b>=0 and y_b<grid_y:
        if state_dict['map'][y_b][x_b][facing_b] == '?' and not get_lidar_c(ranges,state_var):
            if get_lidar_b(ranges,state_var):
                state_dict['map'][y_b][x_b][facing_b] = 1
            else:
                state_dict['map'][y_b][x_b][facing_b] = 0

    x_c = scan_pos[2][0]
    y_c = scan_pos[2][1]
    facing_c = scan_pos[2][2]
    # print(state_dict['map'][y_c][x_c][facing_c])
    if state_dict['map'][y_c][x_c][facing_c] == '?':
        # print("lidar_c",get_lidar_c(ranges,state_var))
        if get_lidar_c(ranges,state_var):
            state_dict['map'][y_c][x_c][facing_c] = 1
        else:
            state_dict['map'][y_c][x_c][facing_c] = 0
    # print(state_dict['map'][y_c][x_c][facing_c])

    x_d = scan_pos[3][0]
    y_d = scan_pos[3][1]
    facing_d = scan_pos[3][2]
    # d can look off map. Check is in map
    if x_d>=0 and x_d<grid_x and y_d>=0 and y_d<grid_y:
        if state_dict['map'][y_d][x_d][facing_d] == '?' and not get_lidar_c(ranges,state_var):
            if get_lidar_d(ranges,state_var):
                state_dict['map'][y_d][x_d][facing_d] = 1
            else:
                state_dict['map'][y_d][x_d][facing_d] = 0

    x_e = scan_pos[4][0]
    y_e = scan_pos[4][1]
    facing_e = scan_pos[4][2]
    if state_dict['map'][y_e][x_e][facing_e] == '?':
        if get_lidar_e(ranges,state_var):
            state_dict['map'][y_e][x_e][facing_e] = 1
        else:
            state_dict['map'][y_e][x_e][facing_e] = 0

    harmonise_map()


def harmonise_map():
    """This function ensures that once a square side has been mapped, it's counterpart
       in the next square is set to be consistent."""

    if state_dict['map'][3][1][1] != '?':
        print("It's here BEFORE!!!!!!",state_dict['map'][3][1][1],state_dict['map'][3][2][3])


    for x in range(0,grid_x):
        for y in range(0,grid_y-1):

            if state_dict['map'][y][x][0] == '?' and state_dict['map'][y+1][x][2] != '?':
                state_dict['map'][y][x][0] = state_dict['map'][y+1][x][2]

            if state_dict['map'][y+1][x][2] == '?' and state_dict['map'][y][x][0] != '?':
                state_dict['map'][y+1][x][2] = state_dict['map'][y][x][0]

    for x in range(0,grid_x-1):
        for y in range(0,grid_y):

            if state_dict['map'][y][x][1] == '?' and state_dict['map'][y][x+1][3] != '?':
                state_dict['map'][y][x][1] = state_dict['map'][y][x+1][3]

            if state_dict['map'][y][x+1][3] == '?' and state_dict['map'][y][x][1] != '?':
                state_dict['map'][y][x+1][3] = state_dict['map'][y][x][1]
    if state_dict['map'][3][1][1] != '?':
        print("It's here AFTER!!!!!!",state_dict['map'][3][1][1],state_dict['map'][3][2][3])




def get_scan_positions():
    """Return indexes of a,b,c,d,e positions from lidar scan, including facing.
       Depends on current position and orientation."""
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    facing = get_facing(state_dict['state_var'])

    if facing == 0:
        facing_a,facing_b,facing_c,facing_d,facing_e = 3,3,0,1,1
        a = (current_x,current_y,facing_a)
        b = (current_x,current_y+1,facing_b)
        c = (current_x,current_y,facing_c)
        d = (current_x,current_y+1,facing_d)
        e = (current_x,current_y,facing_e)
    elif facing == 1:
        facing_a,facing_b,facing_c,facing_d,facing_e = 0,0,1,2,2
        a = (current_x,current_y,facing_a)
        b = (current_x+1,current_y,facing_b)
        c = (current_x,current_y,facing_c)
        d = (current_x+1,current_y,facing_d)
        e = (current_x,current_y,facing_e)
    elif facing == 2:
        facing_a,facing_b,facing_c,facing_d,facing_e = 1,1,2,3,3
        a = (current_x,current_y,facing_a)
        b = (current_x,current_y-1,facing_b)
        c = (current_x,current_y,facing_c)
        d = (current_x,current_y-1,facing_d)
        e = (current_x,current_y,facing_e)
    elif facing == 3:
        facing_a,facing_b,facing_c,facing_d,facing_e = 2,2,3,0,0
        a = (current_x,current_y,facing_a)
        b = (current_x-1,current_y,facing_b)
        c = (current_x,current_y,facing_c)
        d = (current_x-1,current_y,facing_d)
        e = (current_x,current_y,facing_e)

    return a,b,c,d,e

def right_turn(theta=None):
    """First call during turn plan starts a flag and a counter for cycles of turn.
        When counter is finished, turn ends, flag removed and a calibration plan is started."""

    if not state_dict['turn_flag']:
        state_dict['turn_flag'] = True
        if theta is None:
            state_dict['turn_counter'] = TURN_TIME
            state_dict['speed_adj'] = 1
        else:
            state_dict['turn_counter'] = int(2*theta/(np.pi/2) * TURN_TIME)
            state_dict['speed_adj'] = 0.25

    if state_dict['turn_counter'] > 0:
        leftMotor.setVelocity(0.5 * MAX_SPEED * TURN_SPEED_MOD * state_dict['speed_adj'])
        rightMotor.setVelocity(-0.5 * MAX_SPEED * TURN_SPEED_MOD * state_dict['speed_adj'])
        state_dict['turn_counter'] -= 1
    else:
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        if theta is None:
            state_dict['state_var'][2] = format_radians(state_dict['state_var'][2] + np.pi/2)
        else:
            state_dict['state_var'][2] = format_radians(state_dict['state_var'][2] + theta)
        state_dict['turn_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('calibrate_state',None))

def left_turn(theta=None):
    """First call during turn plan starts a flag and a counter for cycles of turn.
        When counter is finished, turn ends, flag removed and a calibration plan is started."""

    if not state_dict['turn_flag']:
        state_dict['turn_flag'] = True
        if theta is None:
            state_dict['turn_counter'] = TURN_TIME
            state_dict['speed_adj'] = 1
        else:
            state_dict['turn_counter'] = int(2*theta/(np.pi/2) * TURN_TIME)
            state_dict['speed_adj'] = 0.25

    if state_dict['turn_counter'] > 0:
        leftMotor.setVelocity(-0.5 * MAX_SPEED * TURN_SPEED_MOD * state_dict['speed_adj'])
        rightMotor.setVelocity(0.5 * MAX_SPEED * TURN_SPEED_MOD * state_dict['speed_adj'])
        state_dict['turn_counter'] -= 1
    else:
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        if theta is None:
            state_dict['state_var'][2] = format_radians(state_dict['state_var'][2] - np.pi/2)
        else:
            state_dict['state_var'][2] = format_radians(state_dict['state_var'][2] - theta)
        state_dict['turn_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('calibrate_state',None))

def forward_correction(dist=None):
    """Needs work to correct position based on real calibrated x,y,f"""
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    current_facing = get_facing(state_dict['state_var'])

    if dist is None:
        dist = 0.5

    if not state_dict['forward_flag']:
        state_dict['forward_flag'] = True
        if dist>=0:
            state_dict['forward_counter'] = int(FORWARD_TIME * dist/0.5)
            state_dict['reverse_indicator'] = 1
        elif dist<0:
            state_dict['forward_counter'] = int(FORWARD_TIME * abs(dist)/0.5)
            state_dict['reverse_indicator'] = -1

    if state_dict['forward_counter'] > 0:
        leftMotor.setVelocity(MAX_SPEED * state_dict['reverse_indicator'])
        rightMotor.setVelocity(MAX_SPEED * state_dict['reverse_indicator'])
        state_dict['forward_counter'] -= 1
    else:
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        state_dict['forward_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('calibrate_state',None))
        if current_facing == 0:
            state_dict['state_var'][1] += 1 * dist/0.5
        elif current_facing == 1:
            state_dict['state_var'][0] += 1 * dist/0.5
        elif current_facing == 2:
            state_dict['state_var'][1] -= 1 * dist/0.5
        elif current_facing == 3:
            state_dict['state_var'][0] -= 1 * dist/0.5



# THIS IS WHERE MOST WORK IS HAPPENING RIGHT NOW!!
def _wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def theta_adjustment_from_points(ideal_pt, observed_pt,
                                 img_width: int = None,
                                 hfov: float = None,     # radians
                                 fx: float = None,       # focal length in pixels
                                 cx: float = None,       # principal point x in pixels
                                 eps: float = 1e-6) -> float:
    """
    Compute yaw error (radians) using ONLY the horizontal pixel coordinate.
    Positive => turn LEFT (CCW), Negative => turn RIGHT (CW).
    """
    ix = float(ideal_pt[0])
    ox = float(observed_pt[0])

    # Derive cx and fx if not provided
    if cx is None and img_width is not None:
        cx = 0.5 * img_width
    if fx is None:
        if hfov is not None and img_width is not None and img_width > 0:
            fx = (0.5 * img_width) / math.tan(0.5 * hfov)
        else:
            raise ValueError("Provide fx, or both img_width and hfov, to compute angle from x.")

    fx = max(fx, eps)
    if cx is None:
        raise ValueError("cx is required (or provide img_width to set cx = img_width/2).")

    # Bearing from x only
    phi_ideal = math.atan((ix - cx) / fx)
    phi_obs   = math.atan((ox - cx) / fx)

    # FIX: use ideal - observed (was observed - ideal)
    return _wrap_to_pi(phi_ideal - phi_obs)


def calibrate_state():
    """use various lidar measures to apply corrections to state variables"""
    # print('start',state_dict['state_var'])
    ranges = lidar.getRangeImage()
    current_facing = get_facing(state_dict['state_var'])

    has_left_wall = False
    has_right_wall = False

    theta_lidar = None   # rad
    theta_cv = None      # rad

    if get_lidar_a(ranges,state_dict['state_var']):

        has_left_wall = True
        start_alpha = (np.pi-2.9)/2
        sample = ranges[0:53:10]
        # alpha is angle relative to -x direction (robot frame)
        alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
        theta = alpha[-1]-alpha[0]
        gamma_expected = np.pi/2 + start_alpha

        # cosine rule
        opposite = (sample[0]**2 + sample[-1]**2 - 2*sample[0]*sample[-1]*np.cos(theta))**0.5

        # sine rule (obtuse angle)
        gamma_actual = np.pi - (np.arcsin(np.sin(theta) * sample[-1] / opposite))

        # print('left',gamma_expected,gamma_actual,gamma_actual - gamma_expected)
        theta_lidar = gamma_actual - gamma_expected
        # state_dict['state_var'][2] += theta_adjustment



    # use right wall to attempt theta correction
    elif get_lidar_e(ranges,state_dict['state_var']):

        has_right_wall = True
        start_alpha = (np.pi-2.9)/2 + 2.6
        sample = ranges[459:512:10]
        # alpha is angle relative to -x direction (robot frame)
        alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
        theta = alpha[-1]-alpha[0]
        gamma_expected = np.pi/2 + (np.pi-2.9)/2

        # cosine rule
        opposite = (sample[0]**2 + sample[-1]**2 - 2*sample[0]*sample[-1]*np.cos(theta))**0.5

        # sine rule (obtuse angle)
        gamma_actual = np.pi - (np.arcsin(np.sin(theta) * sample[0] / opposite))

        # print('right',gamma_expected,gamma_actual,gamma_expected - gamma_actual)
        theta_lidar = gamma_expected - gamma_actual
        # state_dict['state_var'][2] += theta_adjustment

    # use forward left wall
    # elif get_lidar_b(ranges,state_dict['state_var']): -
    # turn off for now with a False condition
    elif ranges[124] < ranges[159] and ranges[159] - ranges[124] < 0.5 and False:

        # print('b',ranges[124],ranges[159])
        start_alpha = (np.pi-2.9)/2 + 0.7
        sample = ranges[124:159:10]
        # alpha is angle relative to -x direction (robot frame)
        alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
        theta = alpha[-1]-alpha[0]
        gamma_expected = np.pi/2 + start_alpha

        # cosine rule
        opposite = (sample[0]**2 + sample[-1]**2 - 2*sample[0]*sample[-1]*np.cos(theta))**0.5

        # sine rule (obtuse angle)
        gamma_actual = np.pi - (np.arcsin(np.sin(theta) * sample[-1] / opposite))

        # print('left',gamma_expected,gamma_actual,gamma_actual - gamma_expected)
        theta_lidar = gamma_actual - gamma_expected
        # state_dict['state_var'][2] += theta_adjustment

    # use forward right
    # elif get_lidar_d(ranges,state_dict['state_var']):
    # turn off for now with a False condition
    elif ranges[353] > ranges[388] and ranges[353] - ranges[388] < 0.5 and False:

        # print('d',ranges[353],ranges[388])
        start_alpha = (np.pi-2.9)/2 + 2
        sample = ranges[353:388:10]
        # alpha is angle relative to -x direction (robot frame)
        alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
        theta = alpha[-1]-alpha[0]
        gamma_expected = np.pi/2 + (np.pi-2.9)/2

        # cosine rule
        opposite = (sample[0]**2 + sample[-1]**2 - 2*sample[0]*sample[-1]*np.cos(theta))**0.5

        # sine rule (obtuse angle)
        gamma_actual = np.pi - (np.arcsin(np.sin(theta) * sample[0] / opposite))

        # print('right',gamma_expected,gamma_actual,gamma_expected - gamma_actual)
        theta_lidar = gamma_expected - gamma_actual
        # state_dict['state_var'][2] += theta_adjustment

    # use forward measurements for theta instead - switched off for now!!! (<0)
    elif max(ranges[256:282:10]) < 0:

        start_alpha = 0
        sample = ranges[256:283:10]
        alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
        theta = alpha[-1]-alpha[0]

        # cosine rule
        opposite = (sample[0]**2 + sample[-1]**2 - 2*sample[0]*sample[-1]*np.cos(theta))**0.5

        gamma_expected = np.pi/2

        if sample[-1] > sample[0]:
        # obtuse angle
            gamma_actual = np.pi - (np.arcsin(np.sin(theta) * sample[0] / opposite))
        elif sample[0] >= sample[-1]:
        # acute angle
            gamma_actual = (np.arcsin(np.sin(theta) * sample[0] / opposite))
        # print('right',gamma_expected,gamma_actual,gamma_expected - gamma_actual)
        theta_lidar = -(gamma_expected - gamma_actual)

        # state_dict['state_var'][2] += theta_adjustment

    print("LIDAR theta", theta_lidar)
    # ---------- CV-based θ ----------
    ideal_nearest_point = (64, 73)        # pixel when robot faces straight
    nearest_point = find_nearest_intersection()
    if nearest_point is not None:
        theta_cv = theta_adjustment_from_points(ideal_nearest_point, nearest_point, img_width=camera.getWidth(), hfov=camera.getFov()
                                                )  # must return radians
        # theta_cv = wrap_angle(theta_cv)
        print("Use CV rotation adjustment", theta_cv)

    # ---------- Kalman fusion ----------
    kf = state_dict['theta_kf']
    dt = TIME_STEP / 1000.0
    kf.predict(dt)

    # Adaptive measurement noise (R) per sensor
    # Tune these baselines to your setup (values below are examples):
    R_CV_BASE = np.deg2rad(0.5)**2     # ~1.5° std when geometry is good
    R_LIDAR_BASE = np.deg2rad(0.5)**2     # ~1.5° std when geometry is good
    R_LIDAR_WEAK = np.deg2rad(1.0)**2     # when geometry is weak

    # LIDAR R
    if theta_lidar is not None:
        R_lidar = R_LIDAR_BASE if (has_left_wall and has_right_wall) else R_LIDAR_WEAK
        kf.update(theta_lidar, R_lidar)

    # CV R
    if theta_cv is not None:
        kf.update(theta_cv, R_CV_BASE)

    # ---------- Apply fused θ once ----------
    theta_adjustment_fused = kf.x
    # Optional: clamp extreme jumps if desired
    theta_adjustment_fused = float(np.clip(theta_adjustment_fused, -np.deg2rad(20), np.deg2rad(20)))
    theta_adjustment = theta_adjustment_fused
    print("Final theta", theta_adjustment)

    state_dict['state_var'][2] += theta_adjustment

    # use forward wall to attempt x,y correction

    if min(ranges[230:282]) < 1:
        min_forward = min(ranges[230:282]) + 0.035 # adding radius of E-Puck
        likely_wall = int(min_forward/0.5)


        # a large correction likely to be a faulty reading so ingore > 0.2
        if current_facing == 0:
            y = state_dict['state_var'][1]
            square_y =int(y)
            wall_measured = square_y + likely_wall
            new_y = wall_measured +1 - min_forward * 2
            forward_adjustment = new_y - state_dict['state_var'][1]
            if abs(forward_adjustment) < 0.5:
                state_dict['state_var'][1] += forward_adjustment

        if current_facing == 1:
            x = state_dict['state_var'][0]
            square_x =int(x)
            wall_measured = square_x + likely_wall
            new_x = wall_measured +1 - min_forward * 2
            forward_adjustment = new_x - state_dict['state_var'][0]
            if abs(forward_adjustment) < 0.5:
                state_dict['state_var'][0] += forward_adjustment

        if current_facing == 2:
            y = state_dict['state_var'][1]
            square_y =int(y)
            wall_measured = square_y - likely_wall
            new_y = wall_measured + min_forward * 2
            forward_adjustment = state_dict['state_var'][1] - new_y
            if abs(forward_adjustment) < 0.5:
                state_dict['state_var'][1] -= forward_adjustment
        if current_facing == 3:
            x = state_dict['state_var'][0]
            square_x =int(x)
            wall_measured = square_x - likely_wall
            new_x = wall_measured + min_forward * 2
            forward_adjustment = state_dict['state_var'][1] - new_x
            if abs(forward_adjustment) < 0.5:
                state_dict['state_var'][0] -= forward_adjustment


    state_dict['goal_stack'].pop(-1)

    if current_facing == 0:
        y = state_dict['state_var'][1]
        square_y =int(y)
        dist_correction = square_y+0.5-y


    elif current_facing == 1:
        x = state_dict['state_var'][0]
        square_x =int(x)
        dist_correction = square_x+0.5-x

    elif current_facing == 2:
        y = state_dict['state_var'][1]
        square_y =int(y)
        dist_correction = y-(square_y+0.5)


    if current_facing == 3:
        x = state_dict['state_var'][0]
        square_x =int(x)
        dist_correction = x-(square_x+0.5)


    if abs(dist_correction) > 0.02 and abs(theta_adjustment) <= 0.05:
    # times 0.5 as converfting squares into distance
        state_dict['goal_stack'].append(('forward_correction',dist_correction*0.5))

    if theta_adjustment > 0.05:
        state_dict['goal_stack'].append(('left_turn',theta_adjustment))
    if theta_adjustment < -0.05:
        state_dict['goal_stack'].append(('right_turn',-theta_adjustment))


def route(x_y):

    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    current_facing = get_facing(state_dict['state_var'])

    if current_x == x_y[0] and current_y == x_y[1]:
        state_dict['goal_stack'].pop(-1)
    else:
        # get next square on route to target
        next_square = get_next_square(x_y)
        state_dict['goal_stack'].append(("move_next",(next_square[0],next_square[1])))


def wavefront_check(x_y,wavefront_squares):
    """helper function to check if a square has already been allocated a key in the wavefront algorithm.
       Returns True if this square has NOT already been mapped with a wavefront."""
    for wavefront in wavefront_squares:
        for square in wavefront_squares[wavefront]:
            if square[0] == x_y[0] and square[1] == x_y[1]:
                return False
    return True


def get_next_square(x_y):

    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])

    wavefront = 1
    wavefront_squares = defaultdict(set)
    wavefront_squares[1].add(x_y)

    map = state_dict['map']

    while True:

        # if wavefront has reached current square, no more waves needed -check here

        for square in wavefront_squares[wavefront]:
            if square[0] == current_x and square[1] == current_y:
                # then identify which of the 4 adjacent squares are accessible as candidate moves
                # i.e. they must not be through a wall
                candidates = []
                # check North
                if map[current_y][current_x][0] == 0:
                    candidates.append((current_x,current_y+1))
                # check East
                if map[current_y][current_x][1] == 0:
                    candidates.append((current_x+1,current_y))
                # check South
                if map[current_y][current_x][2] == 0:
                    candidates.append((current_x,current_y-1))
                # check West

                if map[current_y][current_x][3] == 0:
                    candidates.append((current_x-1,current_y))

                # find a square in the previous wave which is a viable candidate and return
                for candidate in candidates:
                    for lastwave_sq in wavefront_squares[wavefront-1]:
                        if candidate[0] == lastwave_sq[0] and candidate[1] == lastwave_sq[1]:
                            # print("Next square is",candidate[0],candidate[1])
                            return (candidate[0],candidate[1])


        for y in range(grid_y):
            for x in range(grid_x):
                for sq in wavefront_squares[wavefront]:
                    # if square is one east of one in last wavefront AND no wall on east AND not already mapped
                    if x == sq[0] + 1 and y == sq[1] and map[sq[1]][sq[0]][1] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one north of one in last wavefront AND no wall on north AND not already mapped
                    if x == sq[0] and y == sq[1] + 1  and map[sq[1]][sq[0]][0] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one west of one in last wavefront AND no wall on west AND not already mapped
                    if x == sq[0] -1 and y == sq[1]  and map[sq[1]][sq[0]][3] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one south of one in last wavefront AND no wall on south AND not already mapped
                    if x == sq[0] and y == sq[1] -1  and map[sq[1]][sq[0]][2] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
        wavefront += 1


def move_next(x_y):

    """Should be used for a move to an accessible, adjacent square only."""
    # print(state_dict['map'])
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    facing = get_facing(state_dict['state_var'])

    if not state_dict['move_flag']:
        state_dict['move_flag'] = True
        # first turn to face correct direction
        # move to the East
        if x_y[0] > current_x:
            if facing == 2:
                state_dict['goal_stack'].append(("left_turn",None))
            elif facing == 3:
                state_dict['goal_stack'].append(("left_turn",None))
                state_dict['goal_stack'].append(("left_turn",None))
            elif facing == 0:
                state_dict['goal_stack'].append(("right_turn",None))
        # move to the West
        if x_y[0] < current_x:
            if facing == 0:
                state_dict['goal_stack'].append(("left_turn",None))
            elif facing == 1:
                state_dict['goal_stack'].append(("left_turn",None))
                state_dict['goal_stack'].append(("left_turn",None))
            elif facing == 2:
                state_dict['goal_stack'].append(("right_turn",None))
        # move to the North
        if x_y[1] > current_y:
            if facing == 1:
                state_dict['goal_stack'].append(("left_turn",None))
            elif facing == 2:
                state_dict['goal_stack'].append(("left_turn",None))
                state_dict['goal_stack'].append(("left_turn",None))
            elif facing == 3:
                state_dict['goal_stack'].append(("right_turn",None))
        # move to the South
        if x_y[1] < current_y:
            if facing == 3:
                state_dict['goal_stack'].append(("left_turn",None))
            elif facing == 0:
                state_dict['goal_stack'].append(("left_turn",None))
                state_dict['goal_stack'].append(("left_turn",None))
            elif facing == 1:
                state_dict['goal_stack'].append(("right_turn",None))

    else:
        state_dict['move_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('forward_correction',None))

def race(x_y):
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    current_facing = get_facing(state_dict['state_var'])

    if current_x == x_y[0] and current_y == x_y[1]:
        state_dict['goal_stack'].pop(-1)

    else:
        state_dict['goal_stack'].append(("route",(x_y[0],x_y[1])))


def distance_pt(p1, p2):
    """
    Euclidean distance between two 2D points.
    p1, p2: (x, y)
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def seg_len(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def seg_angle_deg(x1, y1, x2, y2):
    # 0 deg = horizontal to the right, returns [0, 180)
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if ang < 0:
        ang += 180.0
    return ang


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def seg_intersection(p1, p2, p3, p4, eps=1e-6):
    """
    Returns the integer pixel (x, y) of the intersection of segments p1-p2 and p3-p4,
    or None if they do not intersect as segments (including colinear-but-disjoint).
    """
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    cx, cy = float(p3[0]), float(p3[1])
    dx, dy = float(p4[0]), float(p4[1])

    r_x, r_y = bx - ax, by - ay
    s_x, s_y = dx - cx, dy - cy
    denom = _cross(r_x, r_y, s_x, s_y)
    if abs(denom) < eps:
        # Parallel
        return None

    t = _cross(cx - ax, cy - ay, s_x, s_y) / denom
    u = _cross(cx - ax, cy - ay, r_x, r_y) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = ax + t * r_x
        iy = ay + t * r_y
        return (int(round(ix)), int(round(iy)))
    return None

def angle_between_deg(a, b):
    """
    Smallest angle between two segments (in degrees, [0, 90]).
    Each segment is a tuple (x1, y1, x2, y2).
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    va = np.array([ax2 - ax1, ay2 - ay1], dtype=float)
    vb = np.array([bx2 - bx1, by2 - by1], dtype=float)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    cosang = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    # Map to [0, 90]
    if ang > 90.0:
        ang = 180.0 - ang
    return ang


def detect_segments_any_angle(edges, min_len=10):
    segs = []

    # Relaxed HoughP to capture short/broken lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=25,       # was 60
        minLineLength=min_len,  # was 25
        maxLineGap=25       # was 15
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if math.hypot(x2-x1, y2-y1) >= min_len:
                segs.append((x1, y1, x2, y2))

    # Optional fallback: LSD (better on faint/broken lines)
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lsd_lines = lsd.detect(edges)[0]
        if lsd_lines is not None:
            for l in lsd_lines.reshape(-1, 4):
                x1, y1, x2, y2 = map(int, l.tolist())
                if math.hypot(x2-x1, y2-y1) >= min_len:
                    segs.append((x1, y1, x2, y2))
    except Exception:
        # LSD may be missing in some builds; safe to ignore
        pass

    # Deduplicate near-duplicates (Hough + LSD overlap)
    dedup = []
    d2 = 4**2  # 4 px end-point tolerance
    for s in segs:
        x1, y1, x2, y2 = s
        keep = True
        for (a1, b1, a2, b2) in dedup:
            if (min((x1-a1)**2 + (y1-b1)**2, (x1-a2)**2 + (y1-b2)**2) <= d2 and
                min((x2-a1)**2 + (y2-b1)**2, (x2-a2)**2 + (y2-b2)**2) <= d2):
                keep = False
                break
        if keep:
            dedup.append(s)
    return dedup


def preprocess(img):
    dilate_kernel = np.ones((3, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mild denoise, then edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120, L2gradient=True)
    edges = cv2.dilate(edges, dilate_kernel, iterations=1)
    return edges


def find_nearest_intersection():
    image_data = camera.getImage()
    img = np.frombuffer(image_data, np.uint8).reshape((height, width, -1))
    img = img[..., :3].copy()

    edges = preprocess(img)

    # Detect segments at any orientation
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=60,
        minLineLength=25, maxLineGap=15
    )

    # Visualize on edge map for clarity
    vis = edges[..., None]
    vis = np.concatenate([vis] * 3, axis=-1)

    # Keep *all* sufficiently long segments (no angle-based filtering)
    min_len = 25  # px
    segments = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if seg_len(x1, y1, x2, y2) >= min_len:
                segments.append((x1, y1, x2, y2))
                ang = seg_angle_deg(x1, y1, x2, y2)
                if abs(ang - 0) < 10 or abs(ang - 180) < 10:
                    color = (0, 255, 0)    # near horizontal
                elif abs(ang - 90) < 10:
                    color = (255, 0, 0)    # near vertical
                else:
                    color = (0, 0, 255)    # other angles
                # cv2.line(vis, (x1, y1), (x2, y2), color, 2)

    # Compute intersections among *all* pairs of (non-parallel) segments
    intersections = []
    dedupe_min_dist = 6  # px
    min_angle_sep_deg = 4.0  # avoid unstable intersections for near-parallel pairs

    def too_close(pt, pts, d=dedupe_min_dist):
        x, y = pt
        for (xx, yy) in pts:
            if (x - xx) * (x - xx) + (y - yy) * (y - yy) <= d * d:
                return True
        return False

    H, W = edges.shape[:2]
    if len(segments) >= 2:
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                a = segments[i]
                b = segments[j]

                # Skip near-parallel pairs
                if angle_between_deg(a, b) < min_angle_sep_deg:
                    continue

                p = seg_intersection((a[0], a[1]), (a[2], a[3]),
                                     (b[0], b[1]), (b[2], b[3]))
                if p is None:
                    continue

                # Keep only points within image bounds
                px, py = p
                if px < 0 or py < 0 or px >= W or py >= H:
                    continue

                if not too_close(p, intersections):
                    intersections.append(p)
                    cv2.circle(vis, p, 4, (0, 255, 255), -1)  # yellow dot

    anchor_point = (64, 128)
    nearest_point = None
    if len(intersections) > 0:
        point_dis = [(distance_pt(point, anchor_point), point) for point in intersections]
        nearest_point = min(point_dis)[1]
        cv2.circle(vis, nearest_point, 4, (0, 0, 255), -1)

    img = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    img_ref = display.imageNew(img.tobytes(), Display.RGB, width, height)
    display.imagePaste(img_ref, 0, 0, False)
    display.imageDelete(img_ref)

    return nearest_point


# behaviours are mapped in this dictionary so the functions can be called with a strong key
behaviours = {"race":race,
              "mapping":mapping,
              "map_current":map_current,
              "right_turn":right_turn,
              "left_turn":left_turn,
              "calibrate_state":calibrate_state,
              "route":route,
              "move_next":move_next,
              "forward_correction":forward_correction}



state = "forward"
goal_stack_string_buffer = "->".join([str(goal) for goal in state_dict['goal_stack']])

while robot.step(TIME_STEP) != -1:

    goal_stack_string = "->".join([str(goal) for goal in state_dict['goal_stack']])
    if goal_stack_string != goal_stack_string_buffer:
        print(goal_stack_string)
        goal_stack_string_buffer = goal_stack_string


    # get current goal plan
    if len(state_dict['goal_stack']) > 0:
        plan = state_dict['goal_stack'][-1]

        # execute current goal plan (tuple of function and arguments - pass arguments if there are any)
        if plan[1]:
            behaviours[plan[0]](plan[1])
        else:
            behaviours[plan[0]]()
    else:
        print("finished")









