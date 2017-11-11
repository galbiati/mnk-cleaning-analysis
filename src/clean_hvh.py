import numpy as np
import pandas as pd
import os
import sys
from collections import OrderedDict as odict

from util.mnkhelpers import bits2boards as b2b

import clean as cl

class HVHData():
    def __init__(self, folder):
        self.original_columns = ['index', 'subject', 'color', 'gi', 'mi', \
                                    'status', 'bp', 'wp', 'response', 'rt', \
                                    'time', 'mouse_t', 'mouse_x']
        self.full_output_columns = ['subject', 'is_comp', 'color', 'status', \
                                    'bp', 'wp', 'response', 'rt', 'gi', 'mi', \
                                    'computer', 'human', 'time', 'a', 'b', \
                                    'aval', 'bval', 'val']
        self.game_model_columns = ['subject', 'color', 'bp', 'wp', 'response', 'rt']


    def load_file(self, folder, file_name, mouse=False):
        """ Initial preparation of data for individual files """

        # load file, drop nuissance columns, remove non-observations
        drop_cols = ['index'] if mouse else ['index', 'mouse_t', 'mouse_x']
        data = pd.read_csv(folder + file_name, names=self.original_columns).drop(drop_cols, axis=1)
        drop_status = (data.status != 'dummy') &  (data.status != 'ready') & (data.status != 'draw offer')
        data = data.loc[drop_status, :].copy().reset_index(drop=True)

        # assign unique subject label (from filename) and create separate cols for humans and computers
        sub_filter = data.rt > 0
        comp_filter = data.rt == 0
        first_move_filter = (data.mi == 0) & (data.gi%2 == 0)
        second_move_filter = (data.mi == 1) & (data.gi%2 == 0)

        data.loc[data.rt > 0, 'subject'] = file_name[:-4]
        data.loc[:, 'human'] = file_name[:-4]
        data.loc[:, 'computer'] = np.nan
        data.loc[comp_filter, 'computer'] = data.loc[comp_filter, 'subject']
        data.loc[first_move_filter, 'computer'] = data.loc[second_move_filter, 'computer']
        data.loc[:, 'computer'] = data.loc[:, 'computer'].fillna(method='ffill')
        data.loc[0, 'computer'] = data.loc[1, 'computer']

        return data





def clean(csv_file, subject_dict):
    cols = ["index", "gi", "mi", "status", "player", "color", "response", "bp", "wp", "rt", "time", "IP"]
    data = pd.read_csv(csv_file, names=cols)
    data = data.loc[(data.status != 'dummy') & (data.status != 'ready') & (data.status != 'draw offer') & (data.status != 'time loss')]
    data.loc[:, "bplen"] = [len(data.loc[i, 'bp']) for i in data.index.values]
    data.loc[:, "wplen"] = [len(data.loc[i, 'wp']) for i in data.index.values]
    data = data.loc[(data.bplen == 36) & (data.wplen == 36) & (data.response != 36)]
    data.mi = data.mi - 1
    data.rt = data.rt / 1000
    cmap = {'B':0, 'W':1}
    smap = {'in progress':'playing', 'win':'win', 'draw':'draw'}
    data["color"] = data.color.map(cmap)
    data["status"] = data.status.map(smap)
    data["subject"] = data.IP.map(subject_dict)
    data = data.drop(["bplen", "wplen", "index", "player", "IP"], axis=1).reset_index(drop=True)
    for i in data.index.values:
        if data.loc[i,'color'] == 0:
            temp = list(data.loc[i,'bp'])
            temp[int(data.loc[i,'response'])] = '0'
            data.loc[i, 'bp'] = "".join(temp)
        else:
            temp = list(data.loc[i,'wp'])
            temp[int(data.loc[i,'response'])] = '0'
            data.loc[i, 'wp'] = "".join(temp)

    reindex_list = ["subject","color","gi","mi","status","bp","wp","response","rt",'time']
    data = data.reindex_axis(reindex_list, axis=1)
    return data
