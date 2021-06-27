'''This is the script that calls to launch the asist_navigation prediction panel'''
import numpy as np
from os import listdir
from os.path import isfile, join
from pandas import (
    DataFrame, HDFStore
)
import pandas as pd
import h5py
import csv
import json
import datetime

from constants import *
from graph_constants_falcon import *

import load_data
import grid2con_graph
import graph_constants_sparky

# distance_path = 'transfer_learning/TL-DCRNN-master/data/input_files/falcon/distances/distances_falcon_hard.csv'
h5_path = 'transfer_learning/TL-DCRNN-master/data/input_files/sparky-0_falcon-e/'
h5_name = 's0fe_s0_triagesignal'
# compare_choice = 'easy_hard' # 'easy_med'
filter_freq = 20 # take 10% of the data, change the density, just as for the testbed

eids = list(ID_TO_REGIONNAME.keys())
for i in range(len(eids)):
    eids[i] = 'fe_' +  eids[i]

# # hardcode for node split for convenience, should make it generic
# mids = list(ID_TO_REGIONNAME.keys())
# special_i = 0
# for i in range(len(mids)):
#     if mids[i] == 'l2':
#         mids[i] = 'l2_0'
#         special_i = i
#     mids[i] = 'm_' +  mids[i]
# mids.insert(special_i+1, 'm_l2_1')
#
# hids = list(ID_TO_REGIONNAME.keys())
# special_k = 0
# for i in range(len(hids)):
#     if hids[i] == 'r2':
#         hids[i] = 'r2_0'
#         special_k = i
#     hids[i] = 'h_' +  hids[i]
# hids.insert(special_k+2, 'h_r2_1')
#
#
# if compare_choice == 'easy_med':
#     ids = eids + mids
# elif compare_choice == 'easy_hard':
#     ids = eids + hids
# print ('ids', ids)
sids = list(graph_constants_sparky.NODE_LOC.keys())
ids = eids + sids




def simulate_single_traj(data_path, trial):
    f = open(data_path, "r")

    current_region = None
    loc_list = []
    for row in f.readlines():
        # print (row)
        # if row != '{' and row != '}':
        #     print (row)
        msg = json.loads(row)
        if 'data' in msg:
            if 'entered_area_name' in msg['data']:
                current_region = msg['data']['entered_area_name']
        if 'header' in msg:
            if 'timestamp' in msg['header']:
                ts = msg['header']['timestamp']
        if ts != 'None' and current_region in graph_constants_sparky.NAME_TO_ID:

            loc_list.append([float(ts), graph_constants_sparky.NAME_TO_ID[current_region]])

    single_trial_graph_feature = []
    ts_list = []
    # filter_freq = 10
    count = 0
    feature = np.zeros(len(ids))
    decay = 0.95
    for tc in sorted(loc_list):
        if count % filter_freq == 0:
            # feature = np.zeros(len(ids))
            feature = feature * decay
            # print (tc)
            feature[ids.index(tc[1])] = 1
            # print (feature)
            single_trial_graph_feature.append(feature)
            ts_value = datetime.datetime(2020, 6, 1+trial) + datetime.timedelta(seconds=tc[0])
            ts_list.append(ts_value)
            # TODO: examine the h5 file that is generated....
        count += 1







    # for ts in ts_list:
    #     print (ts)

    # for f in single_trial_graph_feature:
    #     print (f)


    # positions, perturb, victims, beeps = load_data.read_input(data_path)
    # grid_mats = load_data.make_grid_mat(perturb, victims)
    # adjacent_room = grid2con_graph.find_adjacent_room(perturb)
    # nodes_loc, edges = grid2con_graph.perturbed_graph_from_grid(grid_mats[CON_GRAPH_Y], perturb)

    # write_distances_matrix(nodes_loc, edges)

    # empty_region_count = 0
    #
    #
    # # print ('ids', ids)
    #
    # single_trial_graph_feature = []
    # ts_list = []
    # for t in range(len(positions)): # read the message one by one, paused time ignored
    #     ts, x, z = positions[t]
    #
    #     current_regions = LOC_TO_ID(z - MAP_ZMIN, x - MAP_XMIN_CROPPED)
    #     # print ('ts', ts, 'current_regions', current_regions)
    #     feature = np.zeros(len(ids))
    #     for r in current_regions:
    #         # # temporarily just hard code for the med map node split
    #         # if level == 'm' and r == 'l2':
    #         #     if abs(z - 12) + abs(x - 42) < abs(z - 12) + abs(x - 50):
    #         #         feature[ids.index('m_l2_0')] = 1
    #         #     else:
    #         #         feature[ids.index('m_l2_1')] = 1
    #         #
    #         # elif level == 'h' and r == 'r2':
    #         #     if abs(z - 39) + abs(x - 42) < abs(z - 39) + abs(x - 50):
    #         #         feature[ids.index('h_r2_0')] = 1
    #         #     else:
    #         #         feature[ids.index('h_r2_1')] = 1
    #         #
    #         # else:
    #             feature[ids.index(level +  '_' + r)] = 1
    #     # print (feature)
    #     single_trial_graph_feature.append(feature)
    #     ts_list.append(ts)
    #
    #
    #     if len(current_regions) == 0:
    #         empty_region_count += 1
    #     if empty_region_count > 20:
    #         break
    #     # color_mat = load_data.grid_to_color_map(grid_mats[COLOR_GRID_DISPLAY_Y], x=x, z=z)

    return single_trial_graph_feature, ts_list



def write_features(name, multi_trial_graph_features, multi_trial_ts_list):
    # creating a dataframe, here we use numpy to generate random numbers
    df = DataFrame(multi_trial_graph_features, index=multi_trial_ts_list, columns=ids)

    # creating a HDF5 file
    store = HDFStore(h5_path + name + '.h5')
    # adding dataframe to the HDF5 file
    store.put('df', df, format='table', data_columns=True)

    # viewing the added dataframe
    store['df']

    # append another dataframe to the already existing dataframe in HDF5 file
    # Note: the columns must match otherwise it will raise an ValueError
    # store.append('df', DataFrame(single_trial_graph_feature,  columns=list(ID_TO_REGIONNAME.keys())))
    # to close the HDF5 file
    store.close()

def read_features(name):
    # to open or create a HDF5 file
    store = HDFStore(h5_path + name + '.h5', 'r')

    # reading a HDF5 file
    # this method is not recommended
    # the HDF5 file can store only a single file
    df = pd.read_hdf(h5_path + name + '.h5')

    # to access the dataframe from the HDF5 store
    df = store['df']
    # df1 = store['d2']

    print ('df', df)
    print ('df.shape', df.shape)

    # to close the HDF5 file
    store.close()


if __name__ == '__main__':

    data_folder_path = 'process_human_data/raw_data/huao_collected/keyang_converted/'

    txtfiles = [f for f in listdir(data_folder_path) if isfile(join(data_folder_path, f))]

    multi_trial_graph_features = []
    multi_trial_ts_list = []

    # abbr = {'Easy': 'e', 'Med': 'm', 'Hard': 'h'}
    trial = 0
    for txtfile in txtfiles:
        if '_0' in txtfile:
            data_path = data_folder_path + txtfile
            print('txtfile', txtfile)
            single_trial_graph_feature, ts_list = simulate_single_traj(data_path, trial)
            # print ('len(single_trial_graph_feature)', len(single_trial_graph_feature))
            # print ('len(ts_list)', len(ts_list))
            # print ()
            multi_trial_graph_features += single_trial_graph_feature
            multi_trial_ts_list += ts_list
            trial += 1
            # break


    # print ('len(multi_trial_graph_features)', len(multi_trial_graph_features), 'len(multi_trial_ts_list)', len(multi_trial_ts_list))

    write_features(h5_name, np.array(multi_trial_graph_features), multi_trial_ts_list)
    read_features(h5_name)
