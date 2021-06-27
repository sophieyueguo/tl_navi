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


from constants import *
from graph_constants_falcon import *

import load_data
import grid2con_graph
import graph_constants_sparky

h5_path = 'transfer_learning/TL-DCRNN-master/data/input_files/reverse/s0fm/h5/'
max_trail_count = 10
h5_name = 'testbed_fm_train_' + str(max_trail_count) + 'trials'
# h5_name = 'testbed_fm_test_' + str(max_trail_count) + 'trials'

train_trial = [43, 44, 45, 47, 49, 50, 51, 58, 59, 60, 61, 62, 63, 66, 69, 70, 71, 72, 73, 74, 75, 76, 78, 79, 80, 81, 88, 89, 90, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 118, 119, 121, 122, 123, 129, 131, 132, 134, 135, 136, 140, 141, 142, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 156, 157, 158, 159, 160, 161, 162, 169, 170, 172, 173, 174, 175, 176, 178, 180, 181, 182, 184, 186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 199, 200, 201, 202, 203, 204, 211, 212, 213, 214, 215, 216, 217, 218, 219, 229, 230, 231, 232, 233, 234, 235, 236, 237, 244, 245, 246, 247, 248, 250, 251, 256, 257, 258, 260, 261, 262, 263, 265, 266, 267, 270, 272, 273]
test_trial = [46, 48, 64, 65, 67, 68, 77, 107, 114, 120, 127, 128, 130, 133, 137, 138, 139, 149, 155, 171, 177, 179, 183, 185, 190, 198, 249, 252, 259, 264, 268, 269, 271]
tejus_best_trial = [133, 249, 138, 128, 68, 179, 183, 177, 198, 120]




# eids = list(ID_TO_REGIONNAME.keys())
# for i in range(len(eids)):
#     eids[i] = 'fe_' +  eids[i]

# # hardcode for node split for convenience, should make it generic
mids = list(ID_TO_REGIONNAME.keys())
special_i = 0
for i in range(len(mids)):
    if mids[i] == 'l2':
        mids[i] = 'l2_0'
        special_i = i
    mids[i] = 'fm_' +  mids[i]
mids.insert(special_i+1, 'fm_l2_1')

# hids = list(ID_TO_REGIONNAME.keys())
# special_k = 0
# for i in range(len(hids)):
#     if hids[i] == 'r2':
#         hids[i] = 'r2_0'
#         special_k = i
#     hids[i] = 'fh_' +  hids[i]
# hids.insert(special_k+1, 'fh_r2_1')

#
# if compare_choice == 'easy_med':
#     ids = eids + mids
# elif compare_choice == 'easy_hard':
#     ids = eids + hids
# print ('ids', ids)
sids = list(graph_constants_sparky.NODE_LOC.keys())
ids = sids + mids




def simulate_single_traj(level, data_path, knowledge_condition, vis_window=False):
    positions, perturb, victims, beeps = load_data.read_input(data_path)
    grid_mats = load_data.make_grid_mat(perturb, victims)
    adjacent_room = grid2con_graph.find_adjacent_room(perturb)
    nodes_loc, edges = grid2con_graph.perturbed_graph_from_grid(grid_mats[CON_GRAPH_Y], perturb)

    # write_distances_matrix(nodes_loc, edges)

    empty_region_count = 0


    # print ('ids', ids)

    single_trial_graph_feature = []
    ts_list = []
    feature = np.zeros(len(ids))
    decay = 0.95
    for t in range(len(positions)): # read the message one by one, paused time ignored
        ts, x, z = positions[t]
        feature = feature * decay

        current_regions = LOC_TO_ID(z - MAP_ZMIN, x - MAP_XMIN_CROPPED)
        # print ('ts', ts, 'current_regions', current_regions)
        # feature = np.zeros(len(ids))
        for r in current_regions:
            # temporarily just hard code for the med map node split
            if level == 'm' and r == 'l2':
                if abs(z - 12) + abs(x - 42) < abs(z - 12) + abs(x - 50):
                    feature[ids.index('fm_l2_0')] = 1
                else:
                    feature[ids.index('fm_l2_1')] = 1

            elif level == 'h' and r == 'r2':
                if abs(z - 39) + abs(x - 42) < abs(z - 39) + abs(x - 50):
                    feature[ids.index('fh_r2_0')] = 1
                else:
                    feature[ids.index('fh_r2_1')] = 1

            else:
                feature[ids.index('f' + level +  '_' + r)] = 1
        # print (feature)
        single_trial_graph_feature.append(feature) #TODO: examine features????
        ts_list.append(ts)


        if len(current_regions) == 0:
            empty_region_count += 1
        if empty_region_count > 20:
            break
        # color_mat = load_data.grid_to_color_map(grid_mats[COLOR_GRID_DISPLAY_Y], x=x, z=z)

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

    jsonfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]

    multi_trial_graph_features = []
    multi_trial_ts_list = []

    abbr = {'Easy': 'e', 'Med': 'm', 'Hard': 'h'}

    # for level in ['Easy', 'Med', 'Hard']:
    trial_count = 0
    for level in ['Med']:
        for jsonfile in jsonfiles:
            # if level in jsonfile:
            # if 'Med' in jsonfile and '-TriageSignal' in jsonfile:
            # if 'Trial-46' in jsonfile:
            # if level in jsonfile and '-TriageSignal' in jsonfile:
            # if level in jsonfile and '-TriageSignal' not in jsonfile:
            if level in jsonfile:
                pick = False
                for tr in train_trial:
                # for tr in test_trial:
                     if 'Trial-' + str(tr) in jsonfile:
                         pick = True
                         break
                if pick:
                    trial_count += 1
                    print ('trial_count', trial_count)

                    # break here if we have enough trials
                    if trial_count > max_trail_count:
                        break

                    knowledge_condition = None
                    for kc in ['-NoTriageNoSignal', '-TriageNoSignal', '-TriageSignal']:
                        if kc in jsonfile:
                            knowledge_condition = kc
                            break

                    data_path = DATA_PATH + jsonfile
                    print('jsonfile', jsonfile)
                    single_trial_graph_feature, ts_list = simulate_single_traj(abbr[level], data_path, knowledge_condition, vis_window=False)
                    print ('len(single_trial_graph_feature)', len(single_trial_graph_feature))
                    print ('len(ts_list)', len(ts_list))
                    print ()
                    multi_trial_graph_features += single_trial_graph_feature
                    multi_trial_ts_list += ts_list

            # # # TEMP:
            # break

    print ('len(multi_trial_graph_features)', len(multi_trial_graph_features), 'len(multi_trial_ts_list)', len(multi_trial_ts_list))

    # TEJUS DATA TOO SMALL, DUPLICATE
    # multi_trial_graph_features += multi_trial_graph_features
    # multi_trial_ts_list += multi_trial_ts_list

    write_features(h5_name, np.array(multi_trial_graph_features), multi_trial_ts_list)
    read_features(h5_name)
