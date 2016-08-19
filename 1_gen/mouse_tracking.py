import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""implements Paths object"""

class Paths():
    """Object to organize and hold all paths and plots for a single subject"""
    def __init__(self, data, subject):
        self.size = 100 # size of board squares in pixels
        self.tc = [230,175] # location of top left corner of board
        self.bc = [self.tc[0]+self.size*9, self.tc[1]+self.size*4] # bottom right corner
        self.cX = np.array([self.tc[0] + self.size*i + self.size/2 for i in np.arange(9)])
        self.cY = np.array([self.tc[1] + self.size*i + self.size/2 for i in np.arange(4)])
        xv, yv = np.meshgrid(self.cX, self.cY)
        self.xv = xv.flatten()
        self.yv = yv.flatten()
        self.data = data.loc[data.subject==subject,:]
        self.plots = []
        self.hists = []

    def update_plots(self):
        """Updates self.plots to contain path illustrations for each game"""
        grouped = self.data.groupby(self.data.gi)
        self.plots = np.zeros(len(grouped)).astype(object)
        for name, group in grouped:
            fig, axes = plt.subplots(nrows=(len(group) if len(group)>1 else 2), ncols=1, figsize=(17,10*len(group)))

            for turn in range(len(group)):
                plot_path(group.index.values[turn], axes[turn], self.data)
                plot_endpoint(group.index.values[turn], axes[turn], self.data)
                show_position(group.index.values[turn], axes[turn], self.data, self.xv, self.yv)

            for ax in axes:
                ax.set_xlim([self.tc[0],self.bc[0]])
                ax.set_ylim([self.bc[1],self.tc[1]])
                ax.set_xticks([self.tc[0] + self.size*i for i in np.arange(10)])
                ax.set_yticks([self.tc[1] + self.size*i for i in np.arange(5)])
                ax.set_aspect(1/1)
                ax.set_axis_bgcolor((.9,.9,.9))
                ax.legend()
                ax.grid()

            self.plots[name] = fig
            plt.close(fig)

    def update_hists(self):
        """Updates self.hists to color tiles by time spent hovering over each"""
        grouped = self.data.groupby(self.data.gi)
        self.hists = np.zeros(len(grouped)).astype(object)
        for name, group in grouped:
            fig, axes = plt.subplots(nrows=(len(group) if len(group)>1 else 2), ncols=1, figsize=(17,10*len(group)))

            for turn in range(len(group)):
                print("Nothing here yet!")
                break

# helper funcs

def get_path(I, data):
    if pd.notnull(data.loc[I,"mouseX"]):
        X = data.loc[I,"mouseX"].split(",")
    else:
        X = [0]
    if pd.notnull(data.loc[I,"mouseX"]):
        Y = data.loc[I,"mouseY"].split(",")
    else:
        Y = [0]
    return np.array(list(zip(X,Y)))

def get_coordinates(I, data):
    P = get_path(I, data)
    return (P[-1,0], P[-1,1], data.loc[I,"response"])

def plot_path(I, ax, data):
    P = get_path(I, data)
    ax.plot(P[:,0],P[:,1], label=str(data.loc[I,"response"]))

def plot_endpoint(I, ax, data):
    P = get_path(I, data)
    ax.scatter(P[-1,0], P[-1,1], c='r', s=500, alpha=.75)

def show_position(I, ax, data, xv, yv):
    bp = np.array(list(data.loc[I, "bp"])).astype(bool)
    wp = np.array(list(data.loc[I, "wp"])).astype(bool)
    ax.scatter(xv[bp], yv[bp], marker='o', c='k', s=8000, alpha=.9)
    ax.scatter(xv[wp], yv[wp], marker='o', c='w', s=8000, alpha=.9)
