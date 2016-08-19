import numpy as np
import pandas as pd
import os
import sys
from collections import OrderedDict as odict

from util.mnkhelpers import bits2boards as b2b

# This is almost pretty now! So excited!

class Data():
    """ Data is the primary object for holding experimental data. It also contains functions
        for the loading, cleaning, augmentation, and export of the data tables. """

    def __init__(self, folder):
        self.original_columns = ['index', 'subject', 'color', 'gi', 'mi', \
                                    'status', 'bp', 'wp', 'response', 'rt', \
                                    'time', 'mouse_t', 'mouse_x']
        self.full_output_columns = ['subject', 'is_comp', 'color', 'status', \
                                    'bp', 'wp', 'response', 'rt', 'gi', 'mi', \
                                    'computer', 'human', 'time', 'a', 'b', \
                                    'aval', 'bval', 'val', 'condition', 'group', 'session']
        self.game_model_columns = ['subject', 'color', 'bp', 'wp', 'response', 'rt', \
                                    'condition', 'group', 'session']
        self.afc2_model_columns = ['subject', 'color', 'bp', 'wp', 'response', 'rt', 'a', \
                                   'b', 'aval', 'bval', 'condition', 'group', 'session']
        self.eval_model_columns = ['subject', 'color', 'bp', 'wp', 'response', 'rt', 'val', \
                                    'condition', 'group', 'session']


        self.group_ref = {0: [0, 1, 2, 3, 4],
                          1: [0, 1, 4, 3, 2],
                          2: [2, 1, 0, 3, 4],
                          3: [4, 1, 0, 3, 2],
                          4: [2, 1, 4, 3, 0],
                          5: [4, 1, 2, 3, 0]}
        self.groups = {
            'CE':0, 'GL':1, 'TQ':2, 'SW':3, 'LM':4, 'XW':5, 
            'XM':0, 'XZ':1, 'KL':2, 'MS':3, 'AB':4, 'VC':5,
            'GB':0, 'HS':1, 'YV':2, 'YZ':3, 'EN':4, 'SL':5,
            'IK':0, 'LW':1, 'EW':2, 'AB2':3, 'JC':4, 'OR':5,
            'QC':0, 'JP':1, 'LH':2, 'XG':3, 'SG':4, 'IS':5
       }

        self.group_chart = pd.DataFrame(index=self.groups.keys(), 
                            columns=['a', '2', 'b', '4', 'c'])
        reffer = lambda x: self.group_ref[self.groups[x]]
        for s in self.group_chart.index:
            self.group_chart.loc[s, :] = reffer(s)

        self.objective_errors, self.positions = self.load_boards()
        self.data = self.load(folder)
        self.append_errors(self.data)


    def load_boards(self):
        oe = pd.read_csv('util/objective_errors.csv')
        oe.loc[:, 'bp'] = oe.loc[:, '0_pieces'].map(b2b)
        oe.loc[:, 'wp'] = oe.loc[:, '1_pieces'].map(b2b)
        oe = oe.loc[:, ['bp', 'wp', 'color', 'Game_theoretic_value', 'Confirmed', 'value_Zeyan ', 'confirmed_Zeyan']]
        oe.columns = ['bp', 'wp', 'color', 'gtv', 'gtv_c', 'zv', 'zv_c']

        po_cols = ['bp', 'wp', 'a', 'aval', 'b', 'bval', 'c', 'mu']
        po = pd.read_csv('../1_gen/experiment_boards_new.txt', sep='\t', names=po_cols)
        po.loc[:, 'attempts'] = 0
        po.loc[:, 'errors'] = 0

        return oe, po

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
        first_move_filter = (data.mi == 1) & (data.gi%2 == 0)
        second_move_filter = (data.mi == 2) & (data.gi%2 == 0)

        print(file_name[10:-4])
        data.loc[sub_filter, 'subject'] = file_name[10:-4]
        data.loc[:, 'human'] = file_name[10:-4]
        data.loc[:, 'condition'] = file_name[8]
        data.loc[:, 'group'] = self.groups[file_name[10:-4]]
        print(self.group_chart.loc[file_name[10:-4], file_name[8]])
        data.loc[:, 'session'] = self.group_chart.loc[file_name[10:-4], file_name[8]]
        data.loc[:, 'computer'] = np.nan
        data.loc[comp_filter, 'computer'] = data.loc[comp_filter, 'subject']
        data.loc[first_move_filter, 'computer'] = data.loc[second_move_filter, 'computer']
        data.loc[:, 'computer'] = data.loc[:, 'computer'].fillna(method='ffill')
        data.loc[0, 'computer'] = data.loc[1, 'computer']

        return data

    def load(self, folder):
        """ Calls other functions to corrale data and some support information """
        print('Loading data...')
        self.exp_name = folder
        files = os.listdir(folder + '/Raw/')    
        files = [f for f in files if f[-3:] == 'csv']
        self.subjects = np.unique([f[10:-4] for f in files])
        self.subject_dict = odict(zip(self.subjects, np.arange(len(self.subjects))))
        data = pd.concat([self.load_file(folder + '/Raw/', f) for f in files]).reset_index(drop=True)
        data = self.clean(data)

        return data

    def clean(self, df):
        print('More cleaning...')
        """ Performs further cleaning that can be done on all data collectively """

        # anonymize subjects
        sub_filter = df.rt > 0 # filter computers out
        df.loc[sub_filter, 'subject'] = df.loc[sub_filter, 'subject'].map(self.subject_dict)
        df.loc[:, 'human'] = df.loc[:, 'human'].map(self.subject_dict)

        # give computers identifiable names
        comp_filter = df.rt == 0
        df.loc[comp_filter, 'subject'] = df.loc[comp_filter, 'subject'].astype(int) + 1000
        df.loc[pd.notnull(df.computer), 'computer'] = df.loc[pd.notnull(df.computer), 'computer'].astype(int) + 1000

        # force remove response from board
        for i in df.loc[df.status != 'EVAL', :].index.values:
            if df.loc[i,"color"] == 0:
                l = list(df.loc[i,"bp"])
                l[df.loc[i, "response"]] = '0'
                df.loc[i,"bp"] = ''.join(l)
            else:
                l = list(df.loc[i,"wp"])
                l[df.loc[i,"response"]] = '0'
                df.loc[i,"wp"] = ''.join(l)

        # force correct colors
        count_pieces = lambda x: np.array([np.array(list(df.loc[i, x])).astype(int).sum() for i in df.index.values])
        df.loc[:, 'color'] = count_pieces('bp') - count_pieces('wp')
        df.loc[:, 'color'] = df.loc[:, 'color'].astype(int).astype(str)

        # add is_comp
        is_computer = lambda x: "0" if x > 0 else "1"
        df.loc[:, 'is_comp'] = df.loc[:, 'rt'].map(is_computer)

        # correct move index in games
        df.loc[df.status.isin(['playing', 'win', 'draw']), 'mi'] = df.loc[df.status.isin(['playing', 'win', 'draw']), 'mi'] - 1
        return df

    def append_errors(self, df):
        print('Appending errors...')
        new_cols = ['a', 'b', 'aval', 'bval', 'val']
        for i in new_cols:
            df.loc[:, i] = np.nan
        d = df.loc[:, :]

        for obs in d.index.values:
            bm1 = (self.positions.bp == d.loc[obs, 'bp'])
            bm2 = (self.positions.wp == d.loc[obs, 'wp'])
            board_match_selector = bm1 & bm2
            c = self.positions.loc[board_match_selector, :]

            if len(c) > 1:
                c = int(c.loc[c.index.values[0], :])
            if len(c) == 1:
                for h in new_cols[:-1]:
                    df.loc[obs, h] = int(c.loc[:, h].values[0])

            bm1 = (self.objective_errors.bp == d.loc[obs, 'bp'])
            bm2 = (self.objective_errors.wp == d.loc[obs, 'wp'])
            board_match_selector = bm1 & bm2
            o = self.objective_errors.loc[board_match_selector, :]

            if len(c) > 1:
                o = o.loc[o.index.values[0], :]
            if len(c) == 1:
                df.loc[obs, 'val'] = o.loc[:, 'gtv'].values[0]

        return None

    def export_individuals(self, folder):
        print('Exporting individual trials...')
        for s, i in self.subject_dict.items():
            c = self.data.human == i
            d = self.data.loc[c, :].reset_index(drop=True)
            d = d.reindex_axis(self.full_output_columns, axis=1)
            d.to_csv(folder + '/Clean/' + s + '.csv', index=False)

        return None

    def export_tasks(self, folder):
        print('Exporting by task...')
        # separate tables by task
        game_selector = self.data.loc[:, 'status'].isin(['playing', 'win', 'draw'])
        afc2_selector = self.data.loc[:, 'status'] == 'AFC2'
        eval_selector = self.data.loc[:, 'status'] == 'EVAL'

        game = self.data.loc[game_selector, :]
        afc2 = self.data.loc[afc2_selector, :]
        eva  = self.data.loc[eval_selector, :]
        print(eva.head(5))

        game.loc[:, self.full_output_columns].to_csv(folder + '/Clean/_summaries/all_games_all_fields.csv', index=False)
        game.loc[:, self.game_model_columns].to_csv(folder + '/Clean/_summaries/all_games_model_input.csv', header=False, index=False)
        game.loc[game.rt > 0, self.game_model_columns].to_csv(folder + '/Clean/_summaries/all_games_model_input_no_comp.csv', header=False, index=False)
        afc2.loc[:, self.full_output_columns].to_csv(folder + '/Clean/_summaries/all_afc2s_all_fields.csv', index=False)
        afc2.loc[:, self.afc2_model_columns].to_csv(folder + '/Clean/_summaries/all_afc2s_model_input.csv', header=False, index=False)
        eva.loc[:, self.full_output_columns].to_csv(folder + '/Clean/_summaries/all_evals_all_fields.csv', index=False)
        eva.loc[:, self.eval_model_columns].to_csv(folder + '/Clean/_summaries/all_evals_model_input.csv', header=False, index=False)

        return None

def main(folder):
    D = Data(folder)
    D.export_individuals(folder)
    D.export_tasks(folder)
    return None

if __name__ == '__main__':
    f = sys.argv[1]
    main(f)