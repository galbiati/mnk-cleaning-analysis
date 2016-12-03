import os
import pandas as pd
import numpy as np

e_dir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games/Data/2_eye/New/eyet')
m_dir = os.path.expanduser('~/Google Drive/Bas Zahy Gianni - Games/Data/2_eye/New/mous')

class Tidy(object):
    def __init__(self, initials, sub_no=0):
        self.subject = initials
        self.sub_no = sub_no

        self.tcols = ['tstamp', 'x', 'y', 'dt', 'tile', 'inmove']

        self.gcols = [
            'idx', 'subid', 'color', 'gi', 'mi', 'status',
            'bp', 'wp', 'zet', 'rt', 'tstamp',
            'mouset', 'mousex', 'rt_comp', 'tstamp_comp'
        ]

        self.D = self.load()

        try:
            self.G
        except:
            self.G = self.D
        else:
            pass

        self.T = self.make_tidy()
        self.H = self.make_histogram()

    def load(self):
        pass

    def make_tidy(self):
        pass

    def make_histogram(self):
        H = self.T.pivot_table(
            index='gi', columns='tile', values='dt',
            aggfunc=np.sum, fill_value=0
        )
        H.loc[:, 'rt'] = self.G.loc[H.index, 'rt']
        H.loc[:, 'status'] = self.G.loc[H.index, 'status']
        return H

class Mouse(Tidy):
    def __init__(self, initials, sub_no=0):
        super(Mouse, self).__init__(initials, sub_no)

    def mouse_to_tile(self, x, y):
        top = 192
        bottom = 506
        left = 177
        right = 889
        height = bottom - top
        width = right - left
        newx = 9*(x - left) // width
        newy = 4*(y - top) // height

        return newx + 9*newy

    def load(self):
        d = pd.read_csv('{}/{}.csv'.format(m_dir, self.subject), names=self.gcols)

        d = d.loc[d.rt.astype(int)>0, self.gcols[1:]]
        d.loc[:, 'tstamp_comp'] = d.tstamp.astype(int) - d.rt.astype(int)
        d.loc[:, 'rt_comp'] = d.tstamp_comp - d.tstamp.astype(int).shift(1)
        d.loc[:, 'subject'] = self.sub_no
        d = d.reset_index(drop=True)
        return d

    def make_tidy(self):
        ts = self.D.tstamp.astype(int)
        index = np.arange(ts.max() + 1 - ts.min())
        tidy = pd.DataFrame(index=index, columns=self.tcols)

        tidy.tstamp = np.arange(ts.min(), ts.max()+1, 1)
        tidy.inmove = False
        mt = self.D.loc[~pd.isnull(self.D.mouset), :]

        def expand_line(idx):
            mouset = np.array(mt.loc[idx, 'mouset'].split(',')).astype(int)
            mouset = mouset[mouset > int(mt.tstamp.values[0])]
            iterator = mt.loc[idx, 'mousex'].split(';')
            mousexy = np.array([xy.split(',') for xy in iterator]).astype(int)
            return mouset, mousexy

        for idx in mt.index.values:

            movestart, moveend = mt.loc[idx, ['tstamp_comp', 'tstamp']] - 1
            sel = (tidy.tstamp>=movestart)&(tidy.tstamp<=moveend)
            tidy.loc[sel, 'inmove'] = True
            tidy.loc[sel, 'gi'] = idx
            mouset, mousexy = expand_line(idx)
            tidy.loc[tidy.tstamp.isin(mouset), ['x', 'y']] = mousexy

        tidy = tidy.loc[pd.notnull(tidy.x), :]
        tidy.dt = tidy.tstamp.shift(-1) - tidy.tstamp
        tidy = tidy.loc[tidy.inmove, :]

        tidy.tile = self.mouse_to_tile(tidy.x, tidy.y)

        return tidy

class Eye(Tidy):
    def __init__(self, initials, sub_no=0):
        self.subject = initials
        self.gcols = [
            'idx', 'subid', 'color', 'gi', 'mi', 'status',
            'bp', 'wp', 'zet', 'rt', 'tstamp',
            'mouset', 'mousex', 'rt_comp', 'tstamp_comp'
        ]
        self.G = self.load_games()
        super(Eye, self).__init__(initials, sub_no)

    def load(self):
        eye = pd.read_csv(
            '{}/{}.csv'.format(e_dir, self.subject)
        )

        eye = eye.loc[~pd.isnull(eye.exp_time), :]
        eye.exp_time = (eye.exp_time*1000).astype(int)

        return eye

    def load_games(self):
        d = pd.read_csv(
            '{}/{}.csv'.format(m_dir, self.subject),
            names=self.gcols
        )
        d = d.loc[d.rt.astype(int)>0, self.gcols[1:]]
        d.loc[:, 'tstamp_comp'] = d.tstamp.astype(int) - d.rt.astype(int)
        d.loc[:, 'rt_comp'] = d.tstamp_comp - d.tstamp.astype(int).shift(1)
        d = d.reset_index(drop=True)
        return d

    def make_tidy(self):
        ts = self.D.exp_time.astype(int)
        tidy = pd.DataFrame(
            index=np.arange(ts.max() + 1 - ts.min()),
            columns=self.tcols
        )
        tidy.loc[:, 'tstamp'] = np.arange(ts.min(), ts.max()+1, 1)
        tidy.inmove = False

        tidy.loc[
            tidy.tstamp.isin((1000*self.D.end.values).astype(int)),
            ['x', 'y', 'dt', 'tile']
        ] = self.D.loc[:, ['raw_xloc', 'raw_yloc', 'dur', 'tile']].values
        for idx in self.G.index.values:

            movestart, moveend = self.G.loc[idx, ['tstamp_comp', 'tstamp']] + 1
            sel = (tidy.tstamp>=movestart)&(tidy.tstamp<moveend)
            tidy.loc[sel, 'inmove'] = True
            tidy.loc[sel, 'gi'] = idx

        tidy = tidy.loc[pd.notnull(tidy.x), :]
        tidy.dt = tidy.tstamp.shift(-1) - tidy.tstamp
        tidy = tidy.loc[tidy.inmove, :]

        return tidy
