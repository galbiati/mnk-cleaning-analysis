# packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import scipy.optimize as opt
import seaborn as sns
sns.set_palette("muted")
import re

# custom modules
import data_cleaning as dc

# wish list:

# function to draw lines BETWEEN vertices
# function to determine for each point in which square of grid it lies
# function to show a position from the data and all eye movements overlaid
# function to generate a 2d histogram 'heatmap' on the board showing total duration of fixations on board squares
# function to do the same except as a smoothed/density/contour plot of exact positions

# ^ some of these already in corresponding notebook and just need to be moved over

class Grid():
    """Fits a grid and transform parameters to points"""
    def __init__(self, data, subject):
        self.subject = subject
        self.data = data
        self.initial_vertices = self.make_grid()
        self.vertices = self.initial_vertices.copy()
        self.initial_params = (0, 0, 0, 0, 0, 1, 1)
        self.fit = self.minimize_cost()
        self.transform_params = self.fit.x
        self.vertices = self.transform_grid(self.transform_params)
        self.transformed_data = self.inverse_transform(self.transform_params, self.data)
        self.ilines, self.tlines = self.get_gridlines()
        # self.visual_compare = self.show_grids()
        
    def make_grid(self):
        grid = []
        for x in range(9):
            for y in range(4):
                grid.append([x+.5, y+.5])
        return np.array(grid)
    
    def transform_grid(self, params):
        theta, x, y, v, w, r, s = params
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        trans = np.array([x, y])
        shear = np.array([[1, v], [w, v*w + 1]])
        scale = np.array([r, s])
        rotshear = np.dot(rot, shear)
        return np.dot((self.initial_vertices*scale) + trans, rotshear)
            
    def inverse_transform(self, p, d):
        theta, x, y, v, w, r, s = p
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        trans = np.array([x, y])
        shear = np.array([[1, v], [w, v*w + 1]])
        scale = np.array([r, s])
        rotshear = np.linalg.inv(np.dot(rot, shear))
        return (np.dot(d, rotshear) - trans)/scale


    def to_closest_vertex(self, grid, point):
        return np.linalg.norm(grid - point, axis=1).min()
            
    def cost(self, vals):
        return .5*(vals**2).sum()
    
    def cost_of_grid(self, params, data):
        g = self.transform_grid(params)
        dists = np.array([self.to_closest_vertex(g, i) for i in data])
        return self.cost(dists)
    
    def minimize_cost(self):
        return opt.minimize(self.cost_of_grid, x0=self.initial_params, args=self.data, method="Powell")

    def get_gridlines(self):
        iv = self.initial_vertices
        tv = self.vertices
        ilines = [] # total of 15 lines: 5 'horizontal', 10 'vertical'
        tlines = []
        for j in range(4):
            rrr =  4*np.arange(9) + j
            ilines.append(iv[rrr, :])
            tlines.append(tv[rrr, :])
        for j in range(9):
            ilines.append(iv[(4*j):(4*(j+1)), :])
            tlines.append(tv[(4*j):(4*(j+1)), :])
        return ilines, tlines


    def show_grids(self):
        fig = plt.figure(figsize = (12,7))
        fig.add_subplot(1,1,1)
        ax = fig.axes[0]
        ax.set_title("Initial Guess Vs Fitted Grid - " + self.subject)
        ax.scatter(self.initial_vertices.T[0], self.initial_vertices.T[1], c='blue', s=80, label="Guess", alpha=.3);
        ax.scatter(self.transformed_data.T[0], self.transformed_data.T[1], c='black', s=25, label='Transformed Samples', alpha=.4);
        for i in range(len(self.ilines)):
            ax.plot(self.ilines[i].T[0], self.ilines[i].T[1], color='blue', alpha=.15);
        ax.legend(loc=(1, .5))
        return fig

class Subject():
    """Object that structures, cleans, and processes data for a single subject

    exp_file is the csv file containing experiment data
    eye_file is the asc file containing eyelink data
    subject is a subject ID (ideally the name of the exp/eye files)

    """

    def __init__(self, exp_file, eye_file, subject):
        self.subject = subject
        self.eyelink = self.get_eye_data(eye_file)
        self.experiment = self.get_exp_data(exp_file)
        self.emsg, self.smsg = self.get_eye_msg(eye_file)

        # select and label data
        self.fixations = self.eyelink.loc[self.eyelink.loc[:, 0] == "EFIX", 2:6].astype(float)
        self.fixations.loc[:, 'etype'] = 'F'
        self.fixations.columns = ["etstart", "etend", "dur", "raw_xloc", "raw_yloc", 'etype']
        self.saccs = self.eyelink.loc[self.eyelink.loc[:, 0] == "ESACC", [2,3,4,7,8]].astype(float)
        self.saccs.loc[:, 'etype'] = 'S'
        self.saccs.columns = ["etstart", "etend", "dur", "raw_xloc", "raw_yloc", 'etype'] 
        # self.blinks = self.eyelink.loc[self.eyelink.loc[:, 0] == "EBLINK", [2,3,4,7,8]].astype(float)
        # self.blinks.loc[:, 'etype'] = 'B'
        # self.blinks.columns = ["etstart", "etend", "dur", "raw_xloc", "raw_yloc", 'etype'] 
        self.events = pd.concat([self.fixations, self.saccs], axis=0).sort_index().reset_index(drop=True)
        self.adjust_timestamps()
        self.append_exp_to_fixations()

        # get calibration data
        self.cal = self.get_cal()
        self.cal_coords = self.cal.loc[:, ['raw_xloc', 'raw_yloc']].values
        self.grid = Grid(self.cal_coords, self.subject)
        self.move_plots = []
        self.reverse_transform()

        # divide data up by conditions, by trials, and by moves
        self.conditions = self.divide_by_condition()
        self.games = [self.divide_by_game(c) for c in self.conditions]
        self.moves = [[self.divide_by_move(self.games[c][g]) for g in range(len(self.games[c]))] for c in range(len(self.conditions))]
        self.hists = self.get_gaze_hists()
        

    def get_eye_data(self, filename):
        """Import the eyelink data as pandas dataframe"""
        _ws = ["\t", "    ", "   ", "  "]
        with open(filename, 'r') as f:
            _f = f.read()
            for s in _ws:
                _f = re.sub(s, " ", _f)
            _f = re.sub(" ", "\t", _f)

        with open(filename + "_fix.asc", 'w') as f:
            f.write(_f)

        _data = pd.read_csv(filename + "_fix.asc", skiprows=25, sep="\t", names=range(18), dtype=str)
        return _data

    def get_eye_msg(self, filename):
        """Get the system time and eyelink time from custom MSG"""
        with open(filename, 'r') as f:
            _f = f.readlines()

        _f = [int(i) for i in _f[11][4:-1].split(" ")]
        emsg, smsg = _f[0] / 1000, _f[1:]
        smsg[-1] *= 1000
        smsg = dt(*smsg).timestamp()
        return emsg, smsg

    def get_exp_data(self, filename):
        _d = dc.clean(filename)
        _d.time = _d.time.astype(float) / 1000
        e = _d.loc[_d.status.isin(['win', 'draw', 'playing']) & (_d.computer == _d.subject), :].index
        for ei in e:
            _d.loc[ei, 'time'] = _d.loc[ei+1, 'time'] - _d.loc[ei+1, 'rt']
        return _d

    def adjust_timestamps(self):
        _f = self.events
        _f.loc[:, 'start'] = _f.loc[:, 'etstart'] / 1000 - self.emsg
        _f.loc[:, 'end'] = _f.loc[:, 'etend'] / 1000 - self.emsg
        _f.loc[:, ['start', 'end']] += self.smsg

    def append_exp_to_fixations(self):
        """Append the relevant data from experiment to fixations data"""
        _f = self.events
        _e = self.experiment #.loc[self.experiment.subject==self.experiment.human, :]
        _f.loc[:, 'res'] = np.nan
        _f.loc[:, 'subject'] = self.subject
        _f.loc[:, 'gi'] = np.nan
        _f.loc[:, 'mi'] = np.nan
        _f.loc[:, 'status'] = np.nan
        _f.loc[:, 'exp_time'] = np.nan
        _f.loc[:, 'exp_move_start'] = np.nan
        margin = .001

        for fix in _f.index.values:
            _s = _f.loc[fix, 'start']
            _n = _f.loc[fix, 'end']
            _t = _e.loc[(_e.time >= (_s - margin)) & (_e.time <= (_n + margin)), :]
            if len(_t) > 0:
                _f.loc[fix, 'res'] = _t.response.values[0]
                _f.loc[fix, 'gi'] = _t.gi.values[0]
                _f.loc[fix, 'mi'] = _t.mi.values[0]
                _f.loc[fix, 'status'] = _t.status.values[0]
                _f.loc[fix, 'exp_time'] = _t.time.values[0]
                _f.loc[fix, 'exp_move_start'] = _t.time.values[0] - _t.rt.values[0]
                _f.loc[fix, 'bp'] = _t.bp.values[0]
                _f.loc[fix, 'wp'] = _t.wp.values[0]

    def get_cal(self):
        _f = self.events
        _condition1 = _f.status == "eyecal"
        _condition2 = True # np.abs(self.fixations.time2event) < 2
        _condition3 = (_f.raw_xloc > 0) & (_f.raw_yloc < 1200)
        _condition4 = (_f.raw_yloc > 0) & (_f.raw_yloc < 800)
        return _f.loc[_condition1 & _condition2 & _condition3 & _condition4, :]

    def reverse_transform(self):
        _f = self.events
        d = _f.loc[:, ['raw_xloc', 'raw_yloc']].values
        self.transformed_data = self.grid.inverse_transform(self.grid.transform_params, d)
        _f.loc[:, 'transx'] = self.transformed_data.T[0]
        _f.loc[:, 'transy'] = self.transformed_data.T[1]
        _f.loc[:, 'tile'] = np.floor(_f.loc[:, 'transx'].values).astype(int) + 9*np.floor(_f.loc[:, 'transy'].values).astype(int)
        return None

    def divide_by_condition(self):
        r = self.events
        e = r.loc[r.status=="eyecal", :]
        ei = e.index.values
        start1 = 0
        _dl = np.where(np.diff(ei) > 1000)[0][0]
        end1 = ei[_dl]
        _s2 = r.loc[((r.status=="playing")|(r.status=="win")|(r.status=="draw")), :]
        start2 = _s2.index.values[-1]
        end2 = ei[-1]
        e1 = r.loc[start1:end1, :]
        ai = r.loc[end1+1:start2, :]
        e2 = r.loc[start2+1:end2, :]
        afc = r.loc[end2+1:, :]
        return e1, ai, e2, afc

    def divide_by_game(self, dset):
        games = []
        inds = dset.gi.unique()
        _f = dset.index.values[0]
        for i in range(len(inds))[1:]:
            prind = dset.loc[dset.gi == inds[i-1], :]
            ind = dset.loc[dset.gi == inds[i], :]
            _s = _f
            _e = ind.index.values[-1]
            game = dset.loc[_s:_e, :]
            games.append(game)
            _f = _e + 1
        return games

    def divide_by_move(self, dset):
        #margin = .1
        moves = []
        inds = dset.mi.unique()
        _f = dset.index.values[0]
        for i in range(len(inds))[1:]:
            prind = dset.loc[dset.mi == inds[i-1], :]
            ind = dset.loc[dset.mi == inds[i], :]
            _s = _f
            _e = ind.index.values[-1]
            move = dset.loc[_s:_e, :]
            #move = move.loc[move.start >= move.exp_move_start.values[-1] - margin, :]
            moves.append(move)
            _f = _e +1
        return moves

    def get_gaze_hists(self):
        #margin = .1
        mdfs = []

        for _i in range(len(self.moves)):
            c = self.moves[_i]
            c = [m for g in c for m in g] # down by the bay
            cl = len(c)
            mdf = pd.DataFrame(data=np.zeros([cl, 36]), columns=np.arange(36))
            mdf.loc[:, 'subject'] = np.nan
            mdf.loc[:, 'game'] = np.nan
            mdf.loc[:, 'move'] = np.nan

            for _m in range(cl):
                m = c[_m]
                mdf.loc[_m, 'subject'] = self.subject
                mdf.loc[_m, 'game'] = m.loc[:, 'gi'].values[-1]
                mdf.loc[_m, 'move'] = m.loc[:, 'mi'].values[-1]
                mdf.loc[_m, 'res'] = m.loc[:, 'res'].values[-1]
                mdf.loc[_m, 'bp'] = m.loc[:, 'bp'].values[-1]
                mdf.loc[_m, 'wp'] = m.loc[:, 'wp'].values[-1]
                _probe = m.loc[m.start >= m.exp_move_start.values[-1], :]
                    
                for t in _probe.tile.unique():

                    if t >= 0 and t <= 35:
                        mdf.loc[_m, t] = m.loc[m.tile==t, 'dur'].sum()
                
            mdfs.append(mdf)

        return mdfs