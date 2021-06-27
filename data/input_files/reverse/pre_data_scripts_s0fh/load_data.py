import time
import json
import csv
from datetime import datetime
import numpy as np

from constants import *

def read_input(data_path):
    # only do that after the mission starts
    # ignore the paused period
    positions = []
    start = False
    pause = False

    perturb, victims = [], []

    beeps = []

    record_counter = 0


    with open(data_path) as f:
        for row in f:
            msg = json.loads(row)

            # read the static perturbations and victims
            if msg['topic'] == 'ground_truth/mission/blockages_list':
                for ele in msg['data']['mission_blockage_list']:
                    perturb.append(ele)
            if msg['topic'] == 'ground_truth/mission/victims_list':
                for ele in msg['data']['mission_victim_list']:
                    victims.append(ele)

            # read the potisitons of the player
            if 'data' in msg:
                if 'mission_state' in msg['data']:
                    if msg['data']['mission_state'] == 'Start':
                        start = True
                if start:
                    if not pause:
                        if 'x' in msg['data'] and 'z' in msg['data']:
                            if MAP_YMIN <= int(msg['data']['y']) <= MAP_YMAX:

                                record_counter += 1 # valid data, could be sent, but want to make it less dense
                                if record_counter >= RECORD_PERIOD:
                                    record_counter = 0
                                    ts = time_converter(time_str=msg['header']['timestamp'], time=None, date=None)
                                    positions.append([ts, msg['data']['x'], msg['data']['z']])

                    if 'paused' in msg['data']:
                        pause = msg['data']['paused'] # ignore the pause for convenience, in real time it would be included

            if 'header' in msg:
                msg_ts = 0
                if 'timestamp' in msg['header']:
                    msg_ts = time_converter(time_str=msg['header']['timestamp'], time=None, date=None)
            if 'data' in msg:
                if 'beep_x' in msg['data'] and 'beep_y' in msg['data'] and 'beep_z' in msg['data'] and 'message' in msg['data']:
                    beep_x = msg['data']['beep_x']
                    beep_y = msg['data']['beep_y']
                    beep_z = msg['data']['beep_z']
                    beep_message = msg['data']['message']
                    condition_label = 'before 5 min'
                    if 'mission_timer' in msg['data']:
                        mt = msg['data']['mission_timer']
                        if int(mt[0]) < 5:
                            condition_label = 'after 5 min'
                    beeps.append([msg_ts, beep_x, beep_y, beep_z, beep_message, mt, condition_label])
    # take the second element for sort
    def take_time(elem):
        return elem[0]

    sorted_positions = sorted(positions, key=take_time)
    sorted_beeps = sorted(beeps, key=take_time)

    # print ('sorted_beeps')
    # for row in sorted_beeps:
    #     print (row)

    return sorted_positions, perturb, victims, sorted_beeps



def time_converter(time_str=None, time=None, date=None):
    if time == None and date == None and time_str != None:
        time = time_str.split('T')[1].split('Z')[0]
        date = time_str.split('T')[0]
    elif not(time != None and date != None and time_str == None):
        print ('time converter input wrong')
    if len(time.split('.')) >1:
        date_time_obj = datetime.strptime((date + ',' + time), '%Y-%m-%d,%H:%M:%S.%f')
    else:
        date_time_obj = datetime.strptime((date + ',' + time), '%Y-%m-%d,%H:%M:%S')
    return date_time_obj


def grid_to_color_map(grid_mat, x=None, z=None, tri_v=None):
    if tri_v != None:
        for v in tri_v:
            vj = v[1] - MAP_XMIN
            vi = v[3] - MAP_ZMIN
            grid_mat[vi, vj] = 'air'

    color_mat = np.zeros(grid_mat.shape)
    for i in range(grid_mat.shape[0]):
        for j in range(grid_mat.shape[1]):
            color_mat[i, j] = len(grid_mat[i, j])
            if grid_mat[i, j] in PLOT_COLOR_VAL:
                color_mat[i, j] = PLOT_COLOR_VAL[grid_mat[i, j]] # highlight the victim colors
            else:
                color_mat[i, j] = PLOT_COLOR_VAL['other']
    if x != None and z!= None:
        pj = round(x - MAP_XMIN)
        pi = round(z - MAP_ZMIN)
        if 0 <= pi < color_mat.shape[0] and 0 <= pj < color_mat.shape[1]:
            color_mat[pi, pj] = PLOT_COLOR_VAL['player']


    return color_mat


def make_grid_mat(perturb, victims):
    grid_mats = {}
    for y in range(MAP_YMIN, MAP_YMAX+1):
        grid_mats[y] = np.array([[None] * (MAP_XMAX - MAP_XMIN + 1)] * (MAP_ZMAX - MAP_ZMIN + 1))
        for i in range(MAP_ZMAX - MAP_ZMIN + 1):
            for j in range(MAP_XMAX - MAP_XMIN + 1):
                grid_mats[y][i, j] = 'air'

    with open(BASEMAP_PATH) as f: #load in the basic map
        input = json.loads(f.read())
        for type in input:
            if type == 'doors':
                for row in input[type]:
                    s, e, t, f, o = row
                    x1, y1, z1 = s
                    x2, y2, z2 = e
                    if (x1 != x2) or (z1 != z2):
                        print ('strange door input setting, please check')
                    else:
                        for y in range(y2, y1+1):
                            grid_mats[y][z1 - MAP_ZMIN, x1 - MAP_XMIN] = 'door'
            if type == 'levers' or type == 'data':
                for row in input[type]:
                    if len(row) == 3:
                        p, t, f = row
                        x, y, z = p
                        if grid_mats[y][z - MAP_ZMIN, x - MAP_XMIN] != 'door':
                            grid_mats[y][z - MAP_ZMIN, x - MAP_XMIN] = t
                    elif len(row) == 4:
                        # room wall sign, don't need to include
                        p, t, f, n = row
                    else:
                        print ('please check unknown input data row:', row)

     # load in the perturbations of the environment and the victims
    for p in (perturb + victims):
        x, y, z, block_type = p['x'], p['y'], p['z'], p['block_type']
        x, y, z = int(x), int(y), int(z)
        assert MAP_XMIN <= x <= MAP_XMAX
        assert MAP_YMIN <= y <= MAP_YMAX
        assert MAP_ZMIN <= z <= MAP_ZMAX
        grid_mats[y][z - MAP_ZMIN, x - MAP_XMIN] = block_type

    return grid_mats
