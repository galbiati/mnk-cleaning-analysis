import numpy as np
import pandas as pd
import os

def clean(csv_file, mouse=False):
    """Clean is the main script for cleaning our data"""

    cols = ["index", "subject", "color", "gi", "mi", "status", "bp", "wp", "response", "rt", "time", "mouse_t", "mouse_x"]
    data = pd.read_csv(csv_file, names=cols)
    data.drop(["index","mouse_t","mouse_x"], axis=1, inplace=True)
    data = data.query("status != 'dummy' and status != 'ready' and status != 'draw offer'").copy()
    data = data.reset_index(drop=True)
    data.loc[:, 'rt'] = data.loc[:, 'rt'].astype(float) / 1000

    comp = data.loc[data.loc[:, 'rt'] == 0, 'subject'].unique()
    comp_map = dict(np.array([comp, .01+comp]).T)
    data.loc[data.loc[:, 'subject'] < 1000, 'subject'] = data.loc[data.loc[:, 'subject'] < 1000, 'subject'].map(comp_map)

    inits = data.loc[data.loc[:, 'subject'] >= 1000, 'subject'].unique()
    inits_map = dict(np.array([inits, np.arange(len(inits))]).T)
    data.loc[data.loc[:, 'subject'] >= 1000, 'subject'] = data.loc[data.loc[:, 'subject'].astype(int) >= 1000, "subject"].map(inits_map)
 
    data["human"] = data["subject"]
    data["computer"] = data["subject"]
    for i in np.arange(len(data))[1:]:
        if data.loc[i, 'human'] != int(data.loc[i, 'human']):
            data.loc[i, 'human'] = data.loc[i-1, 'human']
        else:
            data.loc[i, 'computer'] = data.loc[i-1, 'computer']

        if (data.loc[i, 'mi'] == 0) & (i < data.index.values[-1]):
            data.loc[i, 'computer'] = data.loc[i+1,'computer']


    ai = data.loc[((data.status!="AFC2")&(data.status!="EVAL")), :]

    ai.loc[:, 'mi'] = ai.loc[:, 'mi'] - 1
    # ai.loc[:, "color"] = 1 - np.array([np.array(list(data.loc[i, "bp"])).astype(int).sum() for i in data.index.values]) + np.array([np.array(list(data.loc[i, "wp"])).astype(int).sum() for i in data.index.values])

    for i in ai.index.values:
        if ai.loc[i,"color"] == 0:
            l = list(ai.loc[i,"bp"])
            l[ai.loc[i,"response"]] = '0'
            ai.loc[i,"bp"] = ''.join(l)
        else:
            l = list(ai.loc[i,"wp"])
            l[ai.loc[i,"response"]] = '0'
            ai.loc[i,"wp"] = ''.join(l)


    afc = data.loc[data.status=="AFC2", :]
    afc.loc[:, 'color'] = np.array([np.array(list(afc.loc[i, "bp"])).astype(int).sum() for i in afc.index.values]) - np.array([np.array(list(afc.loc[i, "wp"])).astype(int).sum() for i in afc.index.values])
    eva = data.loc[data.status=="EVAL", :]
    eva.loc[:, 'color'] = np.array([np.array(list(eva.loc[i, "bp"])).astype(int).sum() for i in eva.index.values]) - np.array([np.array(list(eva.loc[i, "wp"])).astype(int).sum() for i in eva.index.values])

    data = pd.concat([ai, afc, eva]).reset_index(drop=True)
    reindex_list =  ["human", "computer", "bp", "wp", "response", "status", "subject", "rt", "gi", "mi", "color", 'time']
    data = data.reindex_axis(reindex_list, axis=1)
    return data


def is_computer(rt):
    return "0" if rt > 0 else "1"


subject_names = ["AB", "CE", "GB", "GL", "HS", "KL", "MS", "SW", "TQ", "VC", "XM", "XZ"]


def main():
    subject_dict = dict(zip(subject_names, range(len(subject_names))))
    pd.DataFrame(index=[0], data=subject_dict).to_csv('../Clean/' + 'subject_initials_map.csv')
    for fn in os.listdir('../Raw/')[2:-1]:
        print(fn)
        d = clean('../Raw/' + fn)
        d.loc[:, 'subject'] = subject_dict[fn[-6:-4]]
        d.loc[:, 'is_comp'] = d.loc[:, 'rt'].map(is_computer)
        d.loc[:, 'color'] = d.loc[:, 'color'].astype(int).astype(str)
        # d.loc[:, 'choice a'] = np.nan
        # d.loc[:, 'choice b'] = np.nan

        column_order = ['subject', 'is_comp', 'color', 'status', 
            'bp', 'wp', 'response', 'rt', 'gi', 'mi', 'computer', 'human', 'time']

        d = d.reindex_axis(column_order, axis=1)
        d.to_csv('../Clean/' + fn)

    return None

if __name__=='__main__':
    main()
