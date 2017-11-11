import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sts
from pprint import pprint
import os
from mnkutil import *
sns.set_style('white')
sns.set_style('white')
sns.set_context('poster')
sns.set_palette(['#E97F02', '#490A3D', '#BD1550'])

def get_computer_sequence(df):
    c1 = (df.gi%2 == 0) & (df.mi==2)
    c2 = (df.gi%2 == 1) & (df.mi==3)
    d = df.loc[~df.status.isin(['AFC2', 'EVAL']) & (c1 | c2) ,:]
    return d.computer.values.astype(int)

def append_tables(df):
    for i in new_cols:
        df.loc[:, i] = np.nan
    d = df.loc[df.status=="AFC2", :]
    
    for obs in d.index.values:
        c = positions.loc[
            (positions.bp == d.loc[obs, 'bp']) & (positions.wp == d.loc[obs, 'wp']), 
            :]
        if len(c) > 1:
            c = int(c.loc[c.index.values[0], :])
        if len(c) == 1:
            for h in ['a', 'b', 'aval', 'bval']:
                df.loc[obs, h] = int(c.loc[:, h].values[0])
    
    d = df.loc[df.status=='EVAL', :]
    for obs in d.index.values:
        o = oberr.loc[
            (oberr.bp == d.loc[obs, 'bp']) & (oberr.wp == d.loc[obs, 'wp']), 
            :]
        if len(c) > 1:
            o = o.loc[o.index.values[0], :]
        if len(c) == 1:
            df.loc[obs, 'val'] = o.loc[:, 'gtv'].values[0]
    
    return df

def find_errors(df):
    for i in ['error', 'val']:
        df.loc[:, i] = np.nan
    d = df.loc[df.status=="AFC2", :]
    
    for obs in d.index.values:
        
        c = positions.loc[
            (positions.bp == d.loc[obs, 'bp']) & (positions.wp == d.loc[obs, 'wp']), 
            :]
        o = oberr.loc[
            (oberr.bp == d.loc[obs, 'bp']) & (oberr.wp == d.loc[obs, 'wp']), 
            :]
        
        if len(c) > 1:
            c = c.loc[c.index.values[0], :]
        if len(o) > 1:
            o = o.loc[o.index.values[0], :]
        if len(c) == 1:
            positions.loc[c.index.values[0], 'attempts'] += 1
            if d.loc[obs, 'response'] == c.loc[:, 'a'].values[0]:
                if c.loc[:, 'aval'].values[0] < c.loc[:, 'bval'].values[0]:
                    df.loc[obs, 'error'] = 1
                    positions.loc[c.index.values[0], 'errors'] += 1
                else:
                    df.loc[obs, 'error'] = 0
            elif d.loc[obs, 'response'] == c.loc[:, 'b'].values[0]:
                if c.loc[:, 'aval'].values[0] > c.loc[:, 'bval'].values[0]:
                    df.loc[obs, 'error'] = 1
                    positions.loc[c.index.values[0], 'errors'] += 1
                else:
                    df.loc[obs, 'error'] = 0
        if len(o) == 1:
            df.loc[obs, 'val'] = o.loc[:, 'gtv'].values[0]
    return df

def export_data(ds, save=False):
    gam = pd.concat([d.loc[~d.status.isin(['AFC2', 'EVAL']) & (d.rt != 0), gamout]
                    for d in ds]).reset_index(drop=True)
    afc = pd.concat([d.loc[d.status == 'AFC2', afcout] 
                    for d in ds]).reset_index(drop=True)
    afc.loc[:, new_cols[:-1]] = afc.loc[:, new_cols[:-1]].astype(int)
    eva = pd.concat([d.loc[d.status == 'EVAL', evaout]
                    for d in ds]).reset_index(drop=True)
    eva.loc[:, 'val'] = eva.loc[:, 'val'].astype(int)
    if save:
        gam.to_csv('../Clean/_summaries/all_games.csv', header=False, index=False)
        afc.to_csv('../Clean/_summaries/all_afcs.csv', header=False, index=False)
        eva.to_csv('../Clean/_summaries/all_evals.csv', header=False, index=False)
    
    return gam, afc, eva

def ai_performance(subject_list, subject_data):
    games_x_computer = pd.DataFrame(index=subject_list, columns=list(range(30)), 
                                data=np.zeros([len(subject_list), 30]))
    wins_x_computer = pd.DataFrame(index=subject_list, columns=list(range(30)), 
                               data=np.zeros([len(subject_list), 30]))
    draws_x_computer = pd.DataFrame(index=subject_list, columns=list(range(30)), 
                                data=np.zeros([len(subject_list), 30]))

    for s in subject_list:
        d = subject_data[s]
        w = d.loc[(d.gi%2 == d.mi %2) & (d.status == 'win'), 'computer'].values.astype(int)
        dr = d.loc[(d.gi%2 == d.mi %2) & (d.status == 'draw'), 'computer'].values.astype(int)
        g = get_computer_sequence(d)
        for c in w:
            wins_x_computer.loc[s, c] += 1
        for c in g:
            games_x_computer.loc[s, c] += 1
        for c in dr:
            draws_x_computer.loc[s, c] += 1

    return games_x_computer, wins_x_computer, draws_x_computer

gamout = ['subject', 'color', 'bp', 'wp', 'response', 'rt']
afcout = ['subject', 'color', 'bp', 'wp', 'response', 'rt', 'a', 'b', 'aval', 'bval']
evaout = ['subject', 'color', 'bp', 'wp', 'response', 'rt', 'val']
new_cols = ['a', 'b', 'aval', 'bval', 'val']

oberr = pd.read_csv('../objective_errors.csv')
positions = pd.read_csv('../experiment_boards_new.txt', sep='\t', 
            names=['bp', 'wp', 'a', 'aval', 'b', 'bval', 'c', 'mu'])

oberr.loc[:, 'bp'] = oberr.loc[:, '0_pieces'].map(bits2boards)
oberr.loc[:, 'wp'] = oberr.loc[:, '1_pieces'].map(bits2boards)
oberr = oberr.loc[:, ['bp', 'wp', 'color', 'Game_theoretic_value', 
                      'Confirmed', 'value_Zeyan ', 'confirmed_Zeyan']]
oberr.columns = ['bp', 'wp', 'color', 'gtv', 'gtv_c', 'zv', 'zv_c']
positions.loc[:, 'attempts'] = 0
positions.loc[:, 'errors'] = 0

def main():
    files = [f for f in os.listdir('../Clean/') if ((f[0] != "_") & (f[0] != "."))]
    subjects = [f[:2] if len(f) == 6 else f[:3] for f in files]
    dataset = [pd.read_csv('../Clean/' + f).drop('Unnamed: 0', axis=1) for f in files]
    datadict = dict(zip(subjects, dataset))

    gxc, wxc, dxc = ai_performance(subjects, datadict)
    for df, dfn in list(zip([gxc, wxc, dxc], ['games_x_computer', 'wins_x_computer', 'draws_x_computer'])):
        df.to_csv('../Statistics/' + dfn + '.csv')

    dataset = list(map(append_tables, dataset))
    _, _, _ = export_data(dataset, save=True)

    return None

if __name__ == '__main__':
    main()