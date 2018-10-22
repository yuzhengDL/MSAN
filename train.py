#import cPickle as pkl
import random
import tensorflow as tf
import numpy as np
import math
import datetime
import sys

from collections import OrderedDict, defaultdict

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing

import configuration
import homogeneous_data
from datasets import load_dataset
from vocab import build_dictionary
from model import LTS
import recall

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_dataset_name", "f30k", "name of the training dataset")

def main():
	model_config = configuration.ModelConfig()
	model_config.data = FLAGS.input_dataset_name

	#loading dataset
	print ('Loading %s dataset ...' % FLAGS.input_dataset_name)
	(train_caps, train_ims_local, train_ims_global), (test_caps, test_ims_local, test_ims_global) = load_dataset(name=model_config.data, load_train=True)
	test_vgg_local_feature = test_ims_local
	test_vgg_global_feature = test_ims_global[:, :4096]
	test_NIC_global_feature = test_ims_global[:, 4096:]

	#create and save dictionary
	print ('creating dictionary')
	worddict = build_dictionary(train_caps+test_caps)[0]
	n_words = len(worddict)
	model_config.n_words = n_words
	model_config.worddict = worddict
	print ('dictionary size: ' + str(n_words))
	#with open('f8k.dictionary.pkl', 'wb') as f:
	#	pkl.dump(worddict, f)


	#Building the model
	print ('Building the model ...')
	model = LTS(model_config)
	model.build()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	saver = tf.train.Saver(max_to_keep=model_config.max_checkpoints_to_keep)

	#sess = tf.Session(config=config)

	print('start embedding training')
	curr = 0.
	uidx = 0.
	total_cost = []
	total_r1 = []
	train_iter = homogeneous_data.HomogeneousData(
		data=[train_caps, train_ims_local, train_ims_global],
		batch_size=model_config.batch_size,
		maxlen=model_config.maxlen_w)
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		"""
		model_path = tf.train.latest_checkpoint("checkpoint_files/")
		if model_path:
			print("Loading model from checkpoint: %s", model_path)
			saver.restore(sess, model_path)
			print("Successfully loaded checkpoint: %s", model_path)
		"""

		for epoch in range(model_config.max_epochs):
			# Train G
			print ('Epoch ', epoch)

			if epoch == 15:
				model_config.lrate = model_config.lrate / 10

			for x, im_local, im_global in train_iter:
				uidx += 1

				ls, mask, im_local, im_global = homogeneous_data.prepare_data(
					caps=x,
					features_local=im_local,
					features_global=im_global,
					worddict=worddict,
					maxlen=model_config.maxlen_w,
					n_words=model_config.n_words)

				vgg_global_feature = im_global[:, :4096]
				NIC_feature = im_global[:, 4096:]

				#embedding training
				_, cost = sess.run([model.updates, model.embedding_loss],
								 feed_dict={model.VGG_local_pred_data: im_local,
									    model.VGG_global_pred_data: vgg_global_feature,
									    model.NIC_pred_data: NIC_feature,
											model.ls_pred_data: ls.T,
											model.input_mask: mask.T,
											model.keep_prob: 0.5,
											model.phase: 1,
											model.learning_rate: model_config.lrate})

				#total_cost.append(cost);

				if np.mod(uidx, 10) == 0:
					print ('Epoch ', epoch, 'Update ', uidx, 'Cost ', cost)

				if np.mod(uidx, 100) == 0:
					print('test ...')

					# encode images into the text embedding space
					images = getTestImageFeature(sess, model, test_vgg_local_feature, test_vgg_global_feature, test_NIC_global_feature)
					features = getTestTextFeature(sess, model, model_config, test_caps)

					(r1, r5, r10, medr) = recall.i2t(images, features)
					#total_r1.append(r1)
					print ("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
					(r1i, r5i, r10i, medri) = recall.t2i(images, features)
					print ("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))

					currscore = r1 + r5 + r10 + r1i + r5i + r10i
					if currscore > curr:
						curr = currscore

						# Save model
						print ('Saving...')
						#saver.save(sess, "checkpoint_files/model.ckpt", global_step=int(uidx+1))
						print('done.')

	#total_cost = np.array(total_cost)
	#total_r1 = np.array(total_r1)
	#np.save("cost.npy", total_cost)
	#np.save("r1.npy", total_r1)

	sess = tf.Session()
	model_path = tf.train.latest_checkpoint("checkpoint_files/")
	if not model_path:
		print("Skipping testing. No checkpoint found in: %s", FLAGS.checkpoint_dir)
		return

	print("Loading model from checkpoint: %s", model_path)
	saver.restore(sess, model_path)
	print("Successfully loaded checkpoint: %s", model_path)

	images = getTestImageFeature(sess, model, test_vgg_local_feature, test_NIC_global_feature)

	# encode sentences into the text embedding space
	features = getTestTextFeature(sess, model, model_config, test_caps)

	(r1, r5, r10, medr) = recall.i2t(images, features)
	print ("Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
	(r1i, r5i, r10i, medri) = recall.t2i(images, features)
	print ("Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))



def getTestImageFeature(sess, model, VGG_local_feature, VGG_global_feature, NIC_feature):
	"""Encode images into the text embedding space.

	Inputs:
		sess: current session
		model: model graph
		VGG_feature: VGG features in test dataset
		NIC_feature: NIC features in test dataset

	Output:
		image embedding vectors
	"""
	images = sess.run(model.vfeats_all,
				feed_dict={model.VGG_local_pred_data: VGG_local_feature,
				           model.VGG_global_pred_data: VGG_global_feature,
					   model.NIC_pred_data: NIC_feature,
					   model.keep_prob: 1.0,
					   model.phase: 0})

	bv = images[0].shape[0]
	vfeat_size = images[0].shape[1]
	v_step = len(images)
	vfeats = np.zeros([bv, vfeat_size*v_step])
	for i,vfeat_single in enumerate(images):
		vfeats[:,i*vfeat_size:i*vfeat_size+vfeat_size] = vfeat_single

	return vfeats


def getTestTextFeature(sess, model, model_config, test_caps):
	"""Encode sentences into the text embedding soace

	Inputs:
		sess: current session
		model: current model graph
		model_config: configurations for model parameters
		test_caps: sentences in test dataset

	Output:
		sentence embedding vectors
	"""
	features = np.zeros((len(test_caps), model_config.dim*3), dtype='float32')

	# length dictionary
	ds = defaultdict(list)

	captions = []
	for s in test_caps:
		s = s.lower()
		captions.append(s.split())

	for i,s in enumerate(captions):
		ds[len(s)].append(i)

	#quick check if a word is in the dictionary
	d = defaultdict(lambda : 0)
	for w in model_config.worddict.keys():
		d[w] = 1

	# Get features
	for k in ds.keys():
		numbatches = len(ds[k]) // model_config.batch_size + 1
		for minibatch in range(numbatches):
			caps = ds[k][minibatch::numbatches]
			caption = [captions[c] for c in caps]

			seqs = []
			for i, cc in enumerate(caption):
				seqs.append([model_config.worddict[w] if w in model_config.worddict.keys() else 1 for w in cc])

			x = np.zeros((k+1, len(caption))).astype('int64')
			x_mask = np.zeros((k+1, len(caption))).astype('float32')
			for idx, s in enumerate(seqs):
				x[:k,idx] = s
				x_mask[:k+1,idx] = 1.

			ff = sess.run(model.sfeats_all,
						  feed_dict={model.ls_pred_data: x.T,
									 model.input_mask: x_mask.T,
									 model.keep_prob: 1.0,
									 model.phase: 0})

			bs = ff[0].shape[0]
			sfeat_size = ff[0].shape[1]
			s_step = len(ff)
			sfeats = np.zeros([bs, sfeat_size*s_step])
			for i,sfeat_single in enumerate(ff):
				sfeats[:,i*sfeat_size:i*sfeat_size+sfeat_size] = sfeat_single


			#sfeats = ff[0]

			for ind, c in enumerate(caps):
				features[c] = sfeats[ind]

	return features


if __name__ == '__main__':
	main()
