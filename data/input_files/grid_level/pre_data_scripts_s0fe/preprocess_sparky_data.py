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
h5_path = 'transfer_learning/TL-DCRNN-master/data/input_files/grid_level/s0fe/h5/'
h5_name = 's0'
# compare_choice = 'easy_hard' # 'easy_med'
filter_freq = 20 # take 10% of the data, change the density, just as for the testbed

# sparky
S_XMIN = -2154
S_ZMIN = 150
S_XMAX = -2104
S_ZMAX = 200

# falcon
F_XMIN = -2110
F_ZMIN = 142
F_XMAX = -2020
F_ZMAX = 194

SCALE = 3

ids = []
for f_x in range(F_XMIN, F_XMAX):
    for f_z in range(F_ZMIN, F_ZMAX):
        if f_x % SCALE == 0 and f_z % SCALE == 0:
            ids.append('fx' + str(f_x - F_XMIN) + 'z' + str(f_z - F_ZMIN))

for s_x in range(S_XMIN, S_XMAX):
    for s_z in range(S_ZMIN, S_ZMAX):
        if s_x % SCALE == 0 and s_z % SCALE == 0:
            ids.append('sx' + str(s_x - S_XMIN) + 'z' + str(s_z - S_ZMIN))

print ('ids', ids)
print ('len(ids)', len(ids))


def simulate_single_traj(data_path, trial):
    f = open(data_path, "r")

    current_x, current_z = None, None
    loc_list = []
    for row in f.readlines():
        # print (row)
        # if row != '{' and row != '}':
        #     print (row)
        msg = json.loads(row)
        if 'data' in msg:
            if 'x' in msg['data']:
                current_x = int(msg['data']['x'])
                current_z = int(msg['data']['z'])
        if 'header' in msg:
            if 'timestamp' in msg['header']:
                ts = msg['header']['timestamp']
        if ts != 'None' and current_x != None and current_z != None:
            if S_XMIN <= current_x <= S_XMAX and S_ZMIN <= current_z <= S_ZMAX:
                current_x = current_x - current_x % SCALE - S_XMIN
                current_z = current_z - current_z % SCALE - S_ZMIN
                loc_list.append([float(ts), 'sx' + str(current_x) + 'z' + str(current_z)])

    # for l in loc_list:
    #     print ('l', l)

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
            print ('len(ts_list)', len(ts_list))
            # print ()
            multi_trial_graph_features += single_trial_graph_feature
            multi_trial_ts_list += ts_list
            trial += 1
            # break


    # print ('len(multi_trial_graph_features)', len(multi_trial_graph_features), 'len(multi_trial_ts_list)', len(multi_trial_ts_list))

    write_features(h5_name, np.array(multi_trial_graph_features), multi_trial_ts_list)
    read_features(h5_name)
