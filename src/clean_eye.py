import numpy as np
import pandas as pd

def clean(csv_file, mouse=False):
    """Clean is the main script for cleaning our data"""
    cols = ["index", "subject", "color", "gi", "mi", "status", "bp", "wp", "response", "rt", "time", "mouse_t", "mouse_x"]
    data = pd.read_csv(csv_file, names=cols)
    data.drop(["index","mouse_t","mouse_x"], axis=1, inplace=True)
    data = data.query("status != 'dummy' and status != 'ready' and status != 'draw offer'").copy()
    data.mi = data.mi- 1
    data.rt = data.rt / 1000
    comp_moves = data[data.rt == 0].subject.unique()
    comp_map = dict(np.array([comp_moves, .01+comp_moves]).T)
    data.loc[data["subject"] < 1000, "subject"] = data.loc[data["subject"] < 1000, "subject"].map(comp_map)
    inits = data[data.subject.astype(int) >= 1000].subject.unique()
    inits_map = dict(np.array([inits, np.arange(len(inits))]).T)
    data.loc[data["subject"].astype(int) >= 1000, "subject"] = data.loc[data["subject"].astype(int) >= 1000, "subject"].map(inits_map)
    data["human"] = data["subject"]
    data["computer"] = data["subject"]
    data = data.reset_index(drop=True)
    for i in np.arange(len(data))[1:]:
        if data.loc[i,"human"] != int(data.loc[i,"human"]):
            data.loc[i, "human"] = data.loc[i-1, "human"]
        else:
            data.loc[i, "computer"] = data.loc[i-1, "computer"]
    for i in np.arange(len(data)):
        if data.loc[i,"mi"] == 0:
            data.loc[i,"computer"] = data.loc[i+1,"computer"]
    reindex_list = ["human","computer","bp","wp","response","status","subject","rt","gi","mi","color", "time", "mouseT","mouseX","mouseY"] if mouse else ["human","computer","bp","wp","response","status","subject","rt","gi","mi","color","time"]
    data = data.reindex_axis(reindex_list, axis=1)
    data.loc[:,"color"] = [1 - np.array(list(data.loc[i,"bp"])).astype(int).sum() + np.array(list(data.loc[i,"wp"])).astype(int).sum() for i in data.index.values]
    ai = data.loc[((data.status!="AFC")&(data.status!="EVAL")),:]
    afc = data.loc[data.status=="AFC",:]
    eva = data.loc[data.status=="EVAL",:]
    for i in ai.index.values:
        if ai.loc[i,"color"] == 0:
            l = list(ai.loc[i,"bp"])
            l[ai.loc[i,"response"]] = '0'
            ai.loc[i,"bp"] = ''.join(l)
        else:
            l = list(ai.loc[i,"wp"])
            l[ai.loc[i,"response"]] = '0'
            ai.loc[i,"wp"] = ''.join(l)
    data = pd.concat([ai, afc, eva])
    data = data.reset_index(drop=True)
    return data

def save_data(data, directory="./Clean", name="all", target="model"):
    if target=="model":
        data = data.reindex_axis(["subject","color","gi","mi","status","bp","wp","response","rt","inits"], axis=1)
        data.to_csv(directory + name + "_" + target + ".csv")
    else:
        print("Put what where again?")
