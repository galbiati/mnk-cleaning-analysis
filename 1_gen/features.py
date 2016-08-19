import numpy as np
import pandas as pd

# helper funcs
def expand_boards(d):
	"""Returns all boards in the data as arrays of integers as B, W"""
	x0 = np.array([d.loc[i,"bp"] if d.loc[i,"color"] == "B" else d.loc[i, "wp"] for i in np.arange(len(d))])
	x1 = np.array([d.loc[i,"wp"] if d.loc[i,"color"] == "B" else d.loc[i, "bp"] for i in np.arange(len(d))])
	X0 = np.array([list(i) for i in x0]).astype(np.int)
	X1 = np.array([list(i) for i in x1]).astype(np.int)
	return X0, X1

def expand_feature(f, vsym=None, hsym=None):
	# write this more cleanly with nested for loops or list comps later
	vsym = np.all(np.flipud(f) == f) if vsym==None else vsym
	hsym = np.all(np.fliplr(f) == f) if hsym==None else hsym
	height = f.shape[0]
	width = f.shape[1]
	nhshift = 10 - width
	nvshift = 5 - height
	template = np.zeros([4,9])
	template[0:height, 0:width] = f
	if hsym and vsym:
		Y = np.tile(template.reshape([36]), [nhshift*nvshift,1])
		for i in np.arange(Y.shape[0]):
			Y[i] = np.roll(Y[i], i+(i//nhshift)*(width-1))
	elif hsym and (not vsym):
		template1 = template
		template2 = np.zeros([4,9])
		template2[0:height, 0:width] = np.flipud(f)
		Y1 = np.tile(template1.reshape([36]), [nhshift*nvshift,1])
		Y2 = np.tile(template2.reshape([36]), [nhshift*nvshift,1])
		for i in np.arange(Y1.shape[0]):
			Y1[i] = np.roll(Y1[i], i+(i//nhshift)*(width-1))
		for i in np.arange(Y2.shape[0]):
			Y2[i] = np.roll(Y2[i], i+(i//nhshift)*(width-1))
		Y = np.concatenate((Y1,Y2))
	elif (not hsym) and vsym:
		template1 = template
		template2 = np.zeros([4,9])
		template2[0:height, 0:width] = np.fliplr(f)
		Y1 = np.tile(template1.reshape([36]), [nhshift*nvshift,1])
		Y2 = np.tile(template2.reshape([36]), [nhshift*nvshift,1])
		for i in np.arange(Y1.shape[0]):
			Y1[i] = np.roll(Y1[i], i+(i//nhshift)*(width-1))
		for i in np.arange(Y2.shape[0]):
			Y2[i] = np.roll(Y2[i], i+(i//nhshift)*(width-1))
		Y = np.concatenate((Y1,Y2))
	else:
		if np.all(np.flipud(np.fliplr(f)) == f):
			template1 = template
			template2 = np.zeros([4,9])
			template2[0:height, 0:width] = np.fliplr(f)
			Y1 = np.tile(template1.reshape([36]), [nhshift*nvshift,1])
			Y2 = np.tile(template2.reshape([36]), [nhshift*nvshift,1])
			for i in np.arange(Y1.shape[0]):
				Y1[i] = np.roll(Y1[i], i+(i//nhshift)*(width-1))
			for i in np.arange(Y2.shape[0]):
				Y2[i] = np.roll(Y2[i], i+(i//nhshift)*(width-1))
			Y = np.concatenate((Y1,Y2))
		else:
			template1 = template
			template2 = np.zeros([4,9])
			template2[0:height, 0:width] = np.flipud(f)
			template3 = np.zeros([4,9])
			template3[0:height, 0:width] = np.fliplr(f)
			template4 = np.zeros([4,9])
			template4[0:height, 0:width] = np.flipud(np.fliplr(f))
			Y1 = np.tile(template1.reshape([36]), [nhshift*nvshift,1])
			Y2 = np.tile(template2.reshape([36]), [nhshift*nvshift,1])
			Y3 = np.tile(template3.reshape([36]), [nhshift*nvshift,1])
			Y4 = np.tile(template4.reshape([36]), [nhshift*nvshift,1])
			for i in np.arange(Y1.shape[0]):
				Y1[i] = np.roll(Y1[i], i+(i//nhshift)*(width-1))
			for i in np.arange(Y2.shape[0]):
				Y2[i] = np.roll(Y2[i], i+(i//nhshift)*(width-1))
			for i in np.arange(Y3.shape[0]):
				Y3[i] = np.roll(Y3[i], i+(i//nhshift)*(width-1))
			for i in np.arange(Y4.shape[0]):
				Y4[i] = np.roll(Y4[i], i+(i//nhshift)*(width-1))
			Y = np.concatenate((Y1,Y2,Y3,Y4))
	return Y, vsym, hsym

class Feature(object):
	# feat is a feature coded as a 2d numpy array
	# mask is an 'anti-feature' coded as a 2d numpy array; it must have equal dimensions to feat
	# update: holy crap it works?!
	# no it doesn't...it works well on threats now, but when mask is all zeros there is a problem

	def __init__(self, feat, mask=None):
		self.feature = feat
		self.mask = None if mask is None else mask
		self.feature_matrix, self.vsym, self.hsym = expand_feature(self.feature)
		self.mask_matrix, _, _ = (self.mask, self.mask, self.mask) if self.mask is None else expand_feature(self.mask, vsym=self.vsym, hsym=self.hsym)

	def detect_feature(self, own_p, opp_p):
		self.possession = (np.dot(own_p, self.feature_matrix.T) == self.feature.sum()).astype(int)
		if self.mask != None:
			#np.array([(np.all((i * self.feature_matrix) == self.feature_matrix, axis=1)).astype(int) for i in own_p])
			self.dispossession = (np.dot(opp_p, self.mask_matrix.T) == self.mask.sum()).astype(int) + (np.dot(own_p, self.mask_matrix.T) == self.mask.sum()).astype(int)
			#np.array([(np.all((i * self.mask_matrix) == self.mask_matrix, axis=1)).astype(int) for i in own_p])
			self.dispossession = (self.dispossession * self.possession).astype(int)
			return (self.possession - self.dispossession).sum(axis=1)
		else:
			return self.possession.sum(axis=1)
