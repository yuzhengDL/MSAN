import numpy
import copy
import sys

class HomogeneousData():

	def __init__(self, data, batch_size=128, maxlen=None):
		self.batch_size = 128
		self.data = data
		self.batch_size = batch_size
		self.maxlen = maxlen

		self.prepare()
		self.reset()

	def prepare(self):
		self.caps = self.data[0]
		self.feats_local = self.data[1]
		self.feats_global = self.data[2]

		# find the unique lengths
		self.lengths = [len(cc.split()) for cc in self.caps]
		self.len_unique = numpy.unique(self.lengths)
		# remove any overly long sentences
		if self.maxlen:
			self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]

		# indices of unique lengths
		self.len_indices = dict()
		self.len_counts = dict()
		for ll in self.len_unique:
			self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
			self.len_counts[ll] = len(self.len_indices[ll])

		# current counter
		self.len_curr_counts = copy.copy(self.len_counts)

	def reset(self):
		self.len_curr_counts = copy.copy(self.len_counts)
		self.len_unique = numpy.random.permutation(self.len_unique)
		self.len_indices_pos = dict()
		for ll in self.len_unique:
			self.len_indices_pos[ll] = 0
			self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
		self.len_idx = -1

	def next(self):
		count = 0
		while True:
			self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
			if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
				break
			count += 1
			if count >= len(self.len_unique):
				break
		if count >= len(self.len_unique):
			self.reset()
			raise StopIteration()

		# get the batch size
		curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
		curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
		# get the indices for the current batch
		curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
		self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
		self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size

		caps = [self.caps[ii] for ii in curr_indices]
		feats_local = [self.feats_local[ii//5] for ii in curr_indices]
		feats_global = [self.feats_global[ii//5] for ii in curr_indices]

		return caps, feats_local, feats_global

	def __iter__(self):
		return self

def prepare_data(caps, features_local, features_global, worddict, maxlen=None, n_words=10000):
	"""
	Put data into format useable by the model
	"""
	seqs = []
	local_feat_list = []
	global_feat_list = []
	for i, cc in enumerate(caps):
		cc = cc.lower()
		seqs.append([worddict[w] if w in worddict.keys() else 1 for w in cc.split()])
		local_feat_list.append(features_local[i])
		global_feat_list.append(features_global[i])

	lengths = [len(s) for s in seqs]

	if maxlen != None and numpy.max(lengths) >= maxlen:
		new_seqs = []
		new_feat_list = []
		new_lengths = []
		for l, s, y in zip(lengths, seqs, feat_list):
			if l < maxlen:
				new_seqs.append(s)
				new_feat_list.append(y)
				new_lengths.append(l)
		lengths = new_lengths
		feat_list = new_feat_list
		seqs = new_seqs

		if len(lengths) < 1:
			return None, None, None

	y_local = numpy.zeros((len(local_feat_list), 49, 512)).astype('float16')
	y_global = numpy.zeros((len(global_feat_list), 4608)).astype('float32')
	for idx, ff in enumerate(local_feat_list):
		y_local[idx] = ff
		y_global[idx] = global_feat_list[idx]

	n_samples = len(seqs)
	maxlen = numpy.max(lengths)+1

	x = numpy.zeros((maxlen, n_samples)).astype('int64')
	x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
	for idx, s in enumerate(seqs):
		x[:lengths[idx],idx] = s
		x_mask[:lengths[idx]+1,idx] = 1.

	return x, x_mask, y_local, y_global

