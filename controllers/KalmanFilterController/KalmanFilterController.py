from controller import Robot
import numpy as np
from collections import defaultdict
import json

TIME_STEP = 64 

########## Setup map #########

# Setup map dimensions - 1 grid square is 4 pattern squares (2 brown, 2 beige)
grid_x = 5
grid_y = 5
target_square = (1,4)

grid_size = 0.5 # Measurement of single grid square - should not be changed

# try to read map from file first, otherwise construct empty map
try:
    with open('map.json') as f:
        map = json.load(f)
       
except FileNotFoundError:
    map = [[['?','?','?','?'] for x in range(grid_x)] for y in range(grid_y)]

########## Robot parameters #########

MAX_SPEED = 6.28  # max wheel rotation speed 
TURN_TIME = 14       # steps to turn pi/2
TURN_SPEED_MOD = 0.8 # turn speed adjustment
FORWARD_TIME = 61 # steps for forward one grid square

########## Statistical parameters ##########

# std error for matching wall position when mapping
wall_error = 0.08 

# Precalculated error estimates for Kalman Filter (KF):

# Dead reckoning KF error values estimated by extracting error values with zero sensor noise
dead_reckoning_forward_std_error = 0.03 # (note distance measure, not squares)
dead_reckoning_turn_std_error = 0.08

# Sensor variance for KF
lidar_std_error = 0.003 * 2 # i.e. the sensor 'noise' (0.005) x max_range (2m)
lidar_theta_std_error = 0.06 # estimated numerically for typical side wall theta correction measurement

########## Load robot #########

# load robot
robot = Robot()

# motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# lidar
lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)

########## Setup state variables and flags #########

# All held in a dict for easier access inside functions' scope.
state_dict = {}
state_dict['state_var'] = [0.5,0.5,0.0,0.0,0.0,0.0] # x,y,theta,x_variance,y_variance,theta_variance
state_dict['goal_stack'] = [("race",(target_square[0],target_square[1])),("mapping",None),("calibrate_state",None)]
state_dict['map'] = map
# Flags used for setting turns and moves
state_dict['turn_flag'] = False
state_dict['forward_flag'] = False
state_dict['move_flag'] = False

########## Utility functions #########

def format_radians(theta):
    """Recursive function to keep radian values within range 0 - 2pi
       Assists with identifying 'get_facing' function.
       (Function copied from DP Mid-module Assignment)"""

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
    """ Return robot facing N,E,S,W (0,1,2,3) based on theta nearest """
    theta = format_radians(state_var[2])
    if theta > 7*2*np.pi/8 or theta <= 1*2*np.pi/8:
        return 0
    elif theta > 1*2*np.pi/8 and theta <= 3*2*np.pi/8:
        return 1
    elif theta > 3*2*np.pi/8 and theta <= 5*2*np.pi/8:
        return 2
    else: 
        return 3
        
def get_x_robot_ref(state_var):
    """Detemine x coord axis in robot reference frame.
    Use x axis if facing N or S, y if E or W."""
    facing = get_facing(state_var)
    
    if facing == 0 or facing == 2:
        return state_var[0]
    else:
        return state_var[1]
        
def get_y_robot_ref(state_var):
    """Detemine y coord in robot orientated ref frame.
    Use y axis if facing N or S, x if E or W."""
    facing = get_facing(state_var)
    
    if facing == 0 or facing == 2:
        return state_var[1]
    else:
        return state_var[0]
        
def get_rel_wall_position_left(facing):
    """In combination with the robot_ref functions above, used for determining relative wall position.
       Based on 'lefthand' wall, if facing North or West, the wall will be in a negative axis direction
       relative position to the x coordinate of the robot.  If East or South, it will be positive.
       Same function is used for righthand walls by switching the sign where relevant.
       """
    if facing == 0 or facing == 3:
        return -1
    else:
        return 1
        
def get_rel_wall_position_forward(facing):
    """As for previous function, but for the wall in front.
       If facing North or East, the wall will be in a positive
       relative axis position to the forward coordinate of the robot.  If South or West, it will be negative.
       """
    if facing == 0 or facing == 1:
        return 1
    else:
        return -1
        
def print_maze(maze):
    """Function to pretty print the accumulated map to console."""
    print("___________________________________")
    rows = len(maze)      # grid_y (y: up)
    cols = len(maze[0])   # grid_x (x: across)

    # Print from top row (highest y) to bottom row (y=0)
    for y in range(rows-1, -1, -1):
        visited_chars = []
        for x in range(cols):
            cell = maze[y][x]
            # Visited if all walls are known (not '?')
            if '?' in cell:
                visited_chars.append("?")
            else:
                visited_chars.append("V")
        # Top wall row
        top_row = "|  " + "\t  ".join(str(maze[y][x][0]) for x in range(cols)) + "   |"
        print(top_row)
        # Middle row: left wall, visited, right wall
        middle_parts = []
        for x in range(cols):
            cell = maze[y][x]
            cell_part = f"{cell[3]} {visited_chars[x]} {cell[1]}"
            middle_parts.append(cell_part)
        middle_row = "|" + "\t".join(middle_parts) + " |"
        print(middle_row)
        # Bottom wall row
        bottom_row = "|  " + "\t  ".join(str(maze[y][x][2]) for x in range(cols)) + "   |"
        print(bottom_row)
        if y == 0:
            print("|__________________________________|\n")
        else:
            print("|                           ")

########## lidar wall detection functions ##########

# These five functions identify wall presence in 5 locations relative to the robot
# Each uses a subsection of lidar results to determine an implied wall location

# a - left
# b - forward left
# c - forward
# d - forward right
# e - right
# see report for diagram

# The left and right functions (a,b,d,e) compare impled x coordinate to predicted, 
# The forward function (c) uses y coordinate, all in the robot's reference frame.
# A sum of squares error is taken and compared to a threshold to determine wall presence
    
def get_lidar_a(ranges,state_var):

    # get axis direction information
    facing = get_facing(state_var)
    x_robot_ref = get_x_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_left(facing)
    
    # determine ray angles - full arc is 2.9 radians forwards
    start_alpha = (np.pi-2.9)/2
    sample = ranges[0:53:10] # Use every 10th ray in the arc
    
    # alpha is angle relative to -x direction (robot frame)
    alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
    
    # implied wall location
    sample_x = [x_robot_ref*grid_size + facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    # predicted wall location if one is there
    pred_x = [(x_robot_ref+facing_multiplier*0.5)*grid_size for i in range(len(sample))]
    # sum squares error - compare to wall error tolerance
    x_squared = [(pred_x[i]-sample_x[i])**2 for i in range(len(sample))]
    if sum(x_squared) < len(sample)*wall_error**2:
        return True
    else:
        return False
    
def get_lidar_b(ranges,state_var):

    # get axis direction information
    facing = get_facing(state_var)
    x_robot_ref = get_x_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_left(facing)   
    
    # determine ray angles - full arc is 2.9 radians forwards
    start_alpha = (np.pi-2.9)/2 + 0.7
    sample = ranges[124:159:10]# Use every 10th ray in the arc
    
    # alpha is angle relative to -x direction (robot frame)
    alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
    
    # implied wall location
    sample_x = [x_robot_ref*grid_size + facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    # predicted wall location if one is there
    pred_x = [(x_robot_ref+facing_multiplier*0.5)*grid_size for i in range(len(sample))]
    # sum squares error - compare to wall error tolerance
    x_squared = [(pred_x[i]-sample_x[i])**2 for i in range(len(sample))]
    if sum(x_squared) < len(sample)*wall_error**2:
        return True
    else:
        return False

def get_lidar_c(ranges,state_var):

    # get axis direction information
    facing = get_facing(state_var)
    y_robot_ref = get_y_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_forward(facing) 
    
    # determine ray angles - full arc is 2.9 radians forwards
    start_alpha = (np.pi-2.9)/2 + 1.3
    sample = ranges[230:282:10]# Use every 10th ray in the arc
    # alpha is angle relative to forward direction
    alpha = [3.14/2-(start_alpha+i*10/512*2.9) for i in range(len(sample))]
    
    # implied wall location
    sample_y = [y_robot_ref*grid_size+facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    # predicted wall location if one is there
    pred_y = [(y_robot_ref+facing_multiplier*0.5)*grid_size for i in range(len(sample))]
    # sum squares error - compare to wall error tolerance
    y_squared = [(pred_y[i]-sample_y[i])**2 for i in range(len(sample))]
    if sum(y_squared) < len(sample)*wall_error**2:
        return True
    else:
        return False


def get_lidar_d(ranges,state_var):

    # get axis direction information
    facing = get_facing(state_var)
    x_robot_ref = get_x_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_left(facing)  
    
    # determine ray angles - full arc is 2.9 radians forwards
    start_alpha = (np.pi-2.9)/2 + 2
    sample = ranges[353:388:10]# Use every 10th ray in the arc
    
    # alpha is angle relative to -x direction (robot frame)
    alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
    # implied wall location
    sample_x = [x_robot_ref*grid_size + facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    # predicted wall location if one is there
    pred_x = [(x_robot_ref-facing_multiplier*0.5)*grid_size for i in range(len(sample))]
    # sum squares error - compare to wall error tolerance
    x_squared = [(pred_x[i]-sample_x[i])**2 for i in range(len(sample))]
    if sum(x_squared) < len(sample)*wall_error**2:
        return True
    else:
        return False


def get_lidar_e(ranges,state_var):
    
    # get axis direction information
    facing = get_facing(state_var)
    x_robot_ref = get_x_robot_ref(state_var)
    facing_multiplier = get_rel_wall_position_left(facing)
    
    # determine ray angles - full arc is 2.9 radians forwards
    start_alpha = (np.pi-2.9)/2 + 2.6
    sample = ranges[459:512:10]# Use every 10th ray in the arc
    
    # alpha is angle relative to -x direction (robot frame)
    alpha = [start_alpha+i*10/512*2.9 for i in range(len(sample))]
    # implied wall location
    sample_x = [x_robot_ref*grid_size + facing_multiplier*np.cos(alpha[i])*sample[i] for i in range(len(sample))]
    # predicted wall location if one is there
    pred_x = [(x_robot_ref-facing_multiplier*0.5)*grid_size for i in range(len(sample))]
    # sum squares error - compare to wall error tolerance
    x_squared = [(pred_x[i]-sample_x[i])**2 for i in range(len(sample))]
    if sum(x_squared) < len(sample)*wall_error**2:
        return True
    else:
        return False

########## mapping goal functions ##########

def wavefront_check(x_y,wavefront_squares):
    """Helper function for Wavefront algorithm to check if a square has already been allocated 
        a key in the wavefront algorithm.
       Returns True if this square has NOT already been mapped with a wavefront."""
    for wavefront in wavefront_squares:
        for square in wavefront_squares[wavefront]:
            if square[0] == x_y[0] and square[1] == x_y[1]:
                return False
    return True
       
def check_closest_unmapped_square(x_y,map,unmapped):
    """Use wavefront algorithm to find closest unmapped squares. 
        Wavefront should start from robot's location.
        This function should only be called if there are unmapped squares, so check first."""
    
    # setup wavefront
    wavefront = 1
    wavefront_squares = defaultdict(set)
    # add current square to first wavefront set
    wavefront_squares[1].add(x_y)
    
    while True:
        # if wavefront has already found an unmapped square, just return it as the new target
        for square in wavefront_squares[wavefront]:
            for unmapped_sq in unmapped:
                if square[0] == unmapped_sq[0] and square[1] == unmapped_sq[1]:
                    return (square[0],square[1])
        
        # Otherwise cycle over all grid squares to identify those matching criteria for next wavefront
        for y in range(grid_y):
            for x in range(grid_x):
                for sq in wavefront_squares[wavefront]:
                    # if square is one east of one in last wavefront AND no wall on east AND not already included
                    if x == sq[0] + 1 and y == sq[1] and map[sq[1]][sq[0]][1] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one north of one in last wavefront AND no wall on north AND not already included
                    if x == sq[0] and y == sq[1] + 1  and map[sq[1]][sq[0]][0] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one west of one in last wavefront AND no wall on west AND not already included
                    if x == sq[0] -1 and y == sq[1]  and map[sq[1]][sq[0]][3] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
                    # if square is one south of one in last wavefront AND no wall on south AND not already included
                    if x == sq[0] and y == sq[1] -1  and map[sq[1]][sq[0]][2] == 0 and wavefront_check((x,y),wavefront_squares):
                        wavefront_squares[wavefront+1].add((x,y))
        wavefront += 1
        
    

def check_map(map,state_var):
    """check if unexplored squares, i.e. squares with '?' in map.
       Return False if none, 
       otherwise return closest unmapped square to robot location."""
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


def mapping():
    """High level goal to coordinate mapping."""
    
    # get nearest unmapped square - False if none.
    nearest_unmapped = check_map(state_dict['map'],state_dict['state_var'])
    
    # if none, complete goal and save completed map to file
    if not nearest_unmapped:
        state_dict['goal_stack'].pop(-1)
        with open('map.json','w') as f:
            json.dump(state_dict['map'],f)
    
    # otherwise continue mapping
    else:
        current_x = int(state_dict['state_var'][0])
        current_y = int(state_dict['state_var'][1])
        
        # if current square not fully mapped, add goal to map current square to goal stack
        if nearest_unmapped[0] == current_x and nearest_unmapped[1] == current_y:
            state_dict['goal_stack'].append(("map_current",None))
        
        # otherwise plan route to nearest unmapped square
        else:
            state_dict['goal_stack'].append(("route",nearest_unmapped))
        
        
        
def map_current():
    """Goal plan to map the current square"""

    # call function that performs the scanning
    scan()
    
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    
    # if square fully mapped, remove goal as completed
    if '?' not in state_dict['map'][current_y][current_x]:
        state_dict['goal_stack'].pop(-1)
        
        
    # else keep goal but add a turn goal to complete first
    # this should ensure square is fully mapped 
    else:
        state_dict['goal_stack'].append(('right_turn',None))
        
def get_scan_positions():
    """Function return location and facing indexes of a,b,c,d,e wall positions 
        using current robot location.
       This allows the map to be updated once walls have been scanned.
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
    
def scan():
    """Function to call wall scanning functions and update map."""
    
    # get relative wall positions of a,b,c,d,e scan positions from function
    scan_pos = get_scan_positions()
    
    # get lidar scan results 
    ranges = lidar.getRangeImage()
    
    # get state variables x,y,theta
    state_var = state_dict['state_var']
    
    # Update map for scan a 
    x_a = scan_pos[0][0]
    y_a = scan_pos[0][1]
    facing_a = scan_pos[0][2]
    if state_dict['map'][y_a][x_a][facing_a] == '?':
        if get_lidar_a(ranges,state_var):
            state_dict['map'][y_a][x_a][facing_a] = 1
        else:
            state_dict['map'][y_a][x_a][facing_a] = 0
    
    # Update map for scan b        
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
                
    # Update map for scan c 
    x_c = scan_pos[2][0]
    y_c = scan_pos[2][1]
    facing_c = scan_pos[2][2]
    
    if state_dict['map'][y_c][x_c][facing_c] == '?':
        
        if get_lidar_c(ranges,state_var):
            state_dict['map'][y_c][x_c][facing_c] = 1
        else:
            state_dict['map'][y_c][x_c][facing_c] = 0
    
    # Update map for scan d      
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
    
    # Update map for scan e         
    x_e = scan_pos[4][0]
    y_e = scan_pos[4][1]
    facing_e = scan_pos[4][2]
    if state_dict['map'][y_e][x_e][facing_e] == '?':
        if get_lidar_e(ranges,state_var):
            state_dict['map'][y_e][x_e][facing_e] = 1
        else:
            state_dict['map'][y_e][x_e][facing_e] = 0
    
    # Ensure consistent record of wall presence in adjacent squares   
    harmonise_map()    
    

def harmonise_map():
    """This function ensures that once a square side has been mapped, its unknown counterpart
       in the next square is set to be consistent."""
    
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
  
########## Navigation functions ##########
    
def right_turn(theta=None):
    """First call during new turn starts a flag and a counter determining number of cycles of turn.
        When counter is finished, turn ends, flag removed, goal completed and a calibration goal is added.
        When called with None as argument, should turn 90 degrees (pi/2), otherwise angle in argument (radians)."""

    # start flag and appropriate turn counter for angle requested
    if not state_dict['turn_flag']:
        state_dict['turn_flag'] = True
        if theta is None:
            state_dict['turn_counter'] = TURN_TIME
            state_dict['speed_adj'] = 1
        else:
            state_dict['turn_counter'] = int(2*theta/(np.pi/2) * TURN_TIME)
            state_dict['speed_adj'] = 0.25
    
    # If positive counter remains, keep turning
    if state_dict['turn_counter'] > 0:
        leftMotor.setVelocity(0.5 * MAX_SPEED * TURN_SPEED_MOD * state_dict['speed_adj'])
        rightMotor.setVelocity(-0.5 * MAX_SPEED * TURN_SPEED_MOD * state_dict['speed_adj'])
        state_dict['turn_counter'] -= 1
        
    # Otherwise close turn and update state variables inc variance for Kalman filter
    else:
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        # theta None is a 90 degree turn
        if theta is None:
            # update theta for dead reckoning
            state_dict['state_var'][2] = format_radians(state_dict['state_var'][2] + np.pi/2)
            # update theta variance for dead reckoning
            state_dict['state_var'][5] += dead_reckoning_turn_std_error**2
        else:
            # update theta for dead reckoning
            state_dict['state_var'][2] = format_radians(state_dict['state_var'][2] + theta)
            # update theta variance for dead reckoning
            state_dict['state_var'][5] += theta/(np.pi/2) * dead_reckoning_turn_std_error**2
        
        # Close flag, complete the turn goal and add calibration goal   
        state_dict['turn_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('calibrate_state',None))
        
    
def left_turn(theta=None):
    """First call during new turn starts a flag and a counter determining number of cycles of turn.
        When counter is finished, turn ends, flag removed, goal completed and a calibration goal is added.
        When called with None as argument, should turn 90 degrees (pi/2), otherwise angle in argument (radians)."""

    # start flag and appropriate turn counter for angle requested
    if not state_dict['turn_flag']:
        state_dict['turn_flag'] = True
        if theta is None:
            state_dict['turn_counter'] = TURN_TIME
            state_dict['speed_adj'] = 1
        else:
            state_dict['turn_counter'] = int(2*theta/(np.pi/2) * TURN_TIME)
            state_dict['speed_adj'] = 0.25
    
    # If positive counter remains, keep turning
    if state_dict['turn_counter'] > 0:
        leftMotor.setVelocity(-0.5 * MAX_SPEED * TURN_SPEED_MOD * state_dict['speed_adj'])
        rightMotor.setVelocity(0.5 * MAX_SPEED * TURN_SPEED_MOD * state_dict['speed_adj'])
        state_dict['turn_counter'] -= 1
    
    # Otherwise close turn and update state variables inc variance for Kalman filter
    else:
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        # theta None is a 90 degree turn
        if theta is None:
            # update theta for dead reckoning
            state_dict['state_var'][2] = format_radians(state_dict['state_var'][2] - np.pi/2)
            # update theta variance for dead reckoning
            state_dict['state_var'][5] += dead_reckoning_turn_std_error**2
        else:
            # update theta for dead reckoning
            state_dict['state_var'][2] = format_radians(state_dict['state_var'][2] - theta)
            # update theta variance for dead reckoning
            state_dict['state_var'][5] += theta/(np.pi/2) * dead_reckoning_turn_std_error**2
        
        # Close flag, complete the turn goal and add calibration goal     
        state_dict['turn_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('calibrate_state',None))
        
        
def forward(dist=None):
    """Goal to move forward. First call during new move starts a flag and a counter determining number of cycles to move.
        When counter is finished, move ends, flag removed, goal completed and a calibration goal is added.
        When called with None as argument, should move one square, otherwise distance in argument.
        Dist can be negative for backwards movement. 
        Note each square is 0.5m across so often need to multiply by 2 to get measurement in squares/coordinates."""
        
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    current_facing = get_facing(state_dict['state_var'])
    
    if dist is None:
        dist = 0.5
    
    # start flag and appropriate move counter for distance requested
    # Also set reverse indicator for backwards movement
    if not state_dict['forward_flag']:
        state_dict['forward_flag'] = True
        if dist>=0:
            state_dict['forward_counter'] = int(FORWARD_TIME * dist/0.5) 
            state_dict['reverse_indicator'] = 1
        elif dist<0:
            state_dict['forward_counter'] = int(FORWARD_TIME * abs(dist)/0.5) 
            state_dict['reverse_indicator'] = -1
    
    # If positive counter remains, keep moving   
    if state_dict['forward_counter'] > 0:
        leftMotor.setVelocity(MAX_SPEED * state_dict['reverse_indicator'])
        rightMotor.setVelocity(MAX_SPEED * state_dict['reverse_indicator'])
        state_dict['forward_counter'] -= 1
        
    # Otherwise close move goal and update state variables inc variance for Kalman filter
    else:
        leftMotor.setVelocity(0)
        rightMotor.setVelocity(0)
        state_dict['forward_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('calibrate_state',None))
        # State variable update depends on facing
        if current_facing == 0:
            # update with dead reckoning
            state_dict['state_var'][1] += 1 * 2*dist
            # update forward coordinate variance with dead reckoning (2x to convert from dist to squares)
            state_dict['state_var'][4] += (2*dead_reckoning_forward_std_error)**2
            # update lateral coordinate variance with dead reckoning (from theta error)
            # Important to update this one too as a large lateral error can accumulate over a large move
            state_dict['state_var'][3] += (2*dist * np.sin(state_dict['state_var'][3]))**2
        elif current_facing == 1:
            # update with dead reckoning
            state_dict['state_var'][0] += 1 * 2*dist
            # update forward coordinate variance with dead reckoning (2x to convert from dist to squares)
            state_dict['state_var'][3] += (2*dead_reckoning_forward_std_error)**2
            # update lateral coordinate variance with dead reckoning (from theta error)
            # Important to update this one too as a large lateral error can accumulate over a large move
            state_dict['state_var'][4] += (2*dist * np.sin(state_dict['state_var'][3]))**2
        elif current_facing == 2:
            # update with dead reckoning
            state_dict['state_var'][1] -= 1 * 2*dist
            # update forward coordinate variance with dead reckoning (2x to convert from dist to squares)
            state_dict['state_var'][4] += (2*dead_reckoning_forward_std_error)**2
            # update lateral coordinate variance with dead reckoning (from theta error)
            # Important to update this one too as a large lateral error can accumulate over a large move
            state_dict['state_var'][3] += (2*dist * np.sin(state_dict['state_var'][3]))**2
        elif current_facing == 3:
            # update with dead reckoning
            state_dict['state_var'][0] -= 1 * 2*dist
            # update forward coordinate variance with dead reckoning (2x to convert from dist to squares)
            state_dict['state_var'][3] += (2*dead_reckoning_forward_std_error)**2
            # update lateral coordinate variance with dead reckoning (from theta error)
            # Important to update this one too as a large lateral error can accumulate over a large move
            state_dict['state_var'][4] += (2*dist * np.sin(state_dict['state_var'][3]))**2
    
   
   
def calibrate_state():
    """Use sensor measures to correct state variables with Kalman Filter.
        Initiate further goals to move to correct position if necessary."""
    
    # Get lidar ranges and current facing
    ranges = lidar.getRangeImage()
    current_facing = get_facing(state_dict['state_var'])
    
    # Empty counters for adjustments
    total_theta_adjustment = 0
    Kt = 0
    number_theta_measurements = 0
    forward_adjustment = 0
    
    # First use left wall to attempt theta correction if present
    if get_lidar_a(ranges,state_dict['state_var']):
        
        theta_adjustment = 0
        theta_adjustments = []
        start_alpha = (np.pi-2.9)/2
        sample = ranges[0:53]
       
        # alpha is angle relative to -x direction (robot frame)
        alpha = [start_alpha+i/512*2.9 for i in range(len(sample))]
      
        # Construct n (10 standard) triangles with different lidar rays to estimate angle error 
        # Each triangle will be comprised of two lidar rays plus the wall
        n=10
        for s in range(n):
            
            # Indicies to use for triangle construction
            start = 0+s
            end = -n+s
            
            # Get theta and the expected obtuse triangle angle if bearing were correct
            theta = alpha[end]-alpha[start]
            gamma_expected = np.pi/2 + alpha[start]
        
            # Get actual obtuse angle determined by actual lidar ray lengths.
            # Using cosine rule
            opposite = (sample[start]**2 + sample[end]**2 - 2*sample[start]*sample[end]*np.cos(theta))**0.5
            # Using sine rule (obtuse angle)
            gamma_actual = np.pi - (np.arcsin(np.sin(theta) * sample[end] / opposite))
        
            # Determine and collect implied adjustments (assuming full confidence)
            theta_adjustment = gamma_actual - gamma_expected
            theta_adjustments.append(theta_adjustment)
      
        theta_adjustment = sum(theta_adjustments )/len(theta_adjustments) 
        
        # Accumulate these errors into total (prior total will be zero if this is first wall checked)
        total_theta_adjustment = (total_theta_adjustment*number_theta_measurements + theta_adjustment *n)/(number_theta_measurements + n)
        
        # Keep record of number of measurements included in estimate
        number_theta_measurements += n
        
        
    # Now use right wall to attempt theta correction if present 
    if get_lidar_e(ranges,state_dict['state_var']):
    
        theta_adjustment = 0
        theta_adjustments = []
        start_alpha = (np.pi-2.9)/2 + 2.6
        sample = ranges[459:512]
        
        # alpha is angle relative to -x direction (robot frame)
        alpha = [start_alpha+i/512*2.9 for i in range(len(sample))]
        
        # Construct n (10 standard) triangles with different lidar rays to estimate angle error 
        # Each triangle will be comprised of two lidar rays plus the wall
        n=10
        for s in range(n):
        
            # Indicies to use for triangle construction
            start = 0+s
            end = -n+s
            
            # Get theta and the expected obtuse triangle angle if bearing were correct
            theta = alpha[end]-alpha[start]
            gamma_expected = np.pi/2 + (np.pi-2.9)/2 + alpha[-1] - alpha[end]
            
            # Get actual obtuse angle determined by actual lidar ray lengths.
            # cosine rule
            opposite = (sample[start]**2 + sample[end]**2 - 2*sample[start]*sample[end]*np.cos(theta))**0.5
            
            # sine rule (obtuse angle)
            gamma_actual = np.pi - (np.arcsin(np.sin(theta) * sample[start] / opposite))
            
            # Determine and collect implied adjustments (assuming full confidence)
            theta_adjustment = gamma_expected - gamma_actual
            theta_adjustments.append(theta_adjustment)
            
        theta_adjustment = sum(theta_adjustments )/len(theta_adjustments)    
        
        # Accumulate these errors into total (prior total will be zero if this is first wall checked)
        total_theta_adjustment = (total_theta_adjustment*number_theta_measurements + theta_adjustment *n)/(number_theta_measurements + n)
        
        # Keep record of number of measurements included in estimate
        number_theta_measurements += n
    
    # Kalman Filter - Use determined theta adjustments to determine state variables and variance 
    if number_theta_measurements > 0:
    
        # Kalman gain
        Kt = state_dict['state_var'][5]/(state_dict['state_var'][5] + lidar_theta_std_error**2 /number_theta_measurements)
        
        # Update theta with innovation and Kalman gain and theta variance
        state_dict['state_var'][2] += Kt*total_theta_adjustment 
        state_dict['state_var'][5] = (1-Kt)*state_dict['state_var'][5]
    
    
    # Now use forward wall to attempt x,y correction
    # Ignore if min distance < 1m or any inf results included as unreliable - the robot sometimes wobbles!
    if min(ranges[230:282]) < 1 and float('inf') not in ranges[230:282]:
        
        # Get the shortest forward ray and take sample of 10 with this in the middle
        min_forward = min(ranges[230:282]) 
        index = ranges[230:282].index(min_forward)
        sample = ranges[230+index-5:230+index+5]
        
        # Take average of these 10 and add the robot radius
        min_forward = sum(sample)/len(sample) + 0.035
        
        # Estimate which wall is being scanned to detemine expected distance
        likely_wall = int(min_forward/0.5)
        
        # Determine Kalman gain for each of x and y coords
        Kx = state_dict['state_var'][3]/(state_dict['state_var'][3] + lidar_std_error**2 /10)  
        Ky = state_dict['state_var'][4]/(state_dict['state_var'][4] + lidar_std_error**2 /10)  
        
        # For each facing direction determine implied error
        # If error > 0.5, likely a faulty reading so ignore
        # Update relevant coord with innovation and Kalman Gain and update the variance 
        if current_facing == 0:
            y = state_dict['state_var'][1]
            square_y =int(y)
            wall_measured = square_y + likely_wall
            new_y = wall_measured +1 - min_forward * 2
            forward_adjustment = new_y - state_dict['state_var'][1]
            if abs(forward_adjustment) < 0.5:
                state_dict['state_var'][1] += forward_adjustment*Ky
                state_dict['state_var'][4] = (1-Ky)*state_dict['state_var'][4]
        
        if current_facing == 1:
            x = state_dict['state_var'][0]
            square_x =int(x)
            wall_measured = square_x + likely_wall
            new_x = wall_measured +1 - min_forward * 2
            forward_adjustment = new_x - state_dict['state_var'][0]
            if abs(forward_adjustment) < 0.5:
                state_dict['state_var'][0] += forward_adjustment*Kx   
                state_dict['state_var'][3] = (1-Kx)*state_dict['state_var'][3]     
        
        if current_facing == 2:
            y = state_dict['state_var'][1]
            square_y =int(y)
            wall_measured = square_y - likely_wall
            new_y = wall_measured + min_forward * 2
            forward_adjustment = state_dict['state_var'][1] - new_y
            if abs(forward_adjustment) < 0.5:
                state_dict['state_var'][1] -= forward_adjustment*Ky    
                state_dict['state_var'][4] = (1-Ky)*state_dict['state_var'][4]
                         
        if current_facing == 3:
            x = state_dict['state_var'][0]
            square_x =int(x)
            wall_measured = square_x - likely_wall
            new_x = wall_measured + min_forward * 2
            forward_adjustment = state_dict['state_var'][0] - new_x
            if abs(forward_adjustment) < 0.5:
                state_dict['state_var'][0] -= forward_adjustment*Kx   
                state_dict['state_var'][3] = (1-Kx)*state_dict['state_var'][3]     
                
    # Remove this goal     
    state_dict['goal_stack'].pop(-1)
   
    # Now set additional goal to turn or move to correct position depending on facing
    if current_facing == 0:
        y = state_dict['state_var'][1]
        square_y =int(y)
        dist_correction = square_y+0.5-y
        
        theta = state_dict['state_var'][2]
        theta_correction = format_radians(0 - theta)
        if theta_correction > np.pi:
            theta_correction = theta_correction - 2*np.pi
        
    elif current_facing == 1:
        x = state_dict['state_var'][0]
        square_x =int(x)
        dist_correction = square_x+0.5-x
        
        theta = state_dict['state_var'][2]
        theta_correction = format_radians(np.pi/2 - theta)
        if theta_correction > np.pi:
            theta_correction = theta_correction - 2*np.pi
        
    elif current_facing == 2:
        y = state_dict['state_var'][1]
        square_y =int(y)
        dist_correction = y-(square_y+0.5)
        
        theta = state_dict['state_var'][2]
        theta_correction = format_radians(np.pi - theta)
        if theta_correction > np.pi:
            theta_correction = theta_correction - 2*np.pi
        
    if current_facing == 3:
        x = state_dict['state_var'][0]
        square_x =int(x)
        dist_correction = x-(square_x+0.5)
        
        theta = state_dict['state_var'][2]
        theta_correction = format_radians(3/2*np.pi - theta)
        if theta_correction > np.pi:
            theta_correction = theta_correction - 2*np.pi
        
    # If both theta and dist need correction, only add theta move goal
    # as dist will be corrected in next cycle
    if abs(dist_correction) > 0.02 and abs(theta_correction) <= 0.06:
    # times 0.5 as converting squares into distance
        state_dict['goal_stack'].append(('forward',dist_correction*0.5))
    
    if theta_correction < -0.06:
        state_dict['goal_stack'].append(('left_turn',-theta_correction))
    if theta_correction > 0.06:
        state_dict['goal_stack'].append(('right_turn',theta_correction))  
    
       
def route(x_y):
    """Main route setting goal.
        Check to see if reached target so remove goal, 
        other determine next square on route and set goal to move there."""
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    current_facing = get_facing(state_dict['state_var'])
    
    # If completed remove goal.
    if current_x == x_y[0] and current_y == x_y[1]:
        state_dict['goal_stack'].pop(-1)
    # otehwise get next square on route to target and set goal to move there
    else:   
        next_square = get_next_square(x_y)
        state_dict['goal_stack'].append(("move_next",(next_square[0],next_square[1])))
            

def get_next_square(x_y):
    """Function to determine the next square to move to on route to another target.
        Uses wavefront algorithm emanating from the target."""

    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    
    # Setup wavefront
    wavefront = 1
    wavefront_squares = defaultdict(set)
    wavefront_squares[1].add(x_y)
    
    map = state_dict['map']
    
    while True:
        
        # if wavefront has reached current robot square, no more waves needed
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
                            return (candidate[0],candidate[1])
                                                
        # Otherwise determine next wavefront
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
    """Goal should be used for a move to an accessible, adjacent square only."""
    
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    facing = get_facing(state_dict['state_var'])
    
    # Set a move flag until completed
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
    
    # Add goal to move forward one square 
    else:
        state_dict['move_flag'] = False
        state_dict['goal_stack'].pop(-1)
        state_dict['goal_stack'].append(('forward',None))

       
def race(x_y):
    """Other primary goal - simply sets a Route goal tot he target square"""
    
    current_x = int(state_dict['state_var'][0])
    current_y = int(state_dict['state_var'][1])
    current_facing = get_facing(state_dict['state_var'])
    
    if current_x == x_y[0] and current_y == x_y[1]:
        state_dict['goal_stack'].pop(-1)
        
    else:
        state_dict['goal_stack'].append(("route",(x_y[0],x_y[1])))
        
      
# behaviours are mapped in this dictionary so the functions can be called with a string key
# from the mechanism managing goals
behaviours = {"race":race,
              "mapping":mapping,
              "map_current":map_current,
              "right_turn":right_turn,
              "left_turn":left_turn,
              "calibrate_state":calibrate_state,
              "route":route,
              "move_next":move_next,
              "forward":forward}

# String of goals for printout
goal_stack_string_buffer = "->".join([str(goal) for goal in state_dict['goal_stack']])

# Main loop
while robot.step(TIME_STEP) != -1:
   
    # This simply manages the printout of goals to the screen for illustration
    goal_stack_string = "->".join([str(goal) for goal in state_dict['goal_stack']])
    if goal_stack_string != goal_stack_string_buffer:
        print_maze(state_dict['map'])
        print(goal_stack_string)
        goal_stack_string_buffer = goal_stack_string
    
    
    # get current goal plan and execute next goal
    if len(state_dict['goal_stack']) > 0:
        plan = state_dict['goal_stack'][-1]
        
        # execute current goal plan (tuple of function and arguments - pass arguments if there are any)
        if plan[1]:
            behaviours[plan[0]](plan[1])
        else:
            behaviours[plan[0]]()
    else:
        print("finished")









