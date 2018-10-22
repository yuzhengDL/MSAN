from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

class LTS:
	"""Cross-media retrieval base on https://doi.org/10.1007/978-3-319-64689-3_24.

	"Learning a Limited Text Space for Cross-Media Retrieval"
	Zheng Yu, Wenmin Wang, Mengdi Fan
	"""


	def __init__(self, options):
		"""Basic setup

		Args:
			options: Object containing configuration parameters.
		"""

		self.config = options

		self.initializer = tf.truncated_normal_initializer(
			mean=0.0, stddev=0.1)

		# Dropout rate
		self.keep_prob = tf.placeholder(tf.float32, name="dropout")
		# Training phase
		self.phase = tf.placeholder(tf.bool, name="traning_phase")
		self.learning_rate = tf.placeholder(tf.float32, name='lrate')

		# Text space embedding vectors fot image features.
		#self.IncV4_embedding = None
		self.image_embedding= None
		self.NIC_embedding = None

		# Text space embedding vectors for sentences.
		self.outputs = None
		self.final_state = None
		self.text_embedding = None

		self.memories_v={}
		self.memories_u={}
		self.avn={}
		self.aut={}
		self.vfeats_all=[]
		self.sfeats_all=[]
		self.sim=None
		self.img_context=None
		self.ls_context=None

		# The embedding loss for the optimizer to optimize.
		self.embedding_loss = None

		# The update operation to optimize the embedding loss
		self.updates = None

	def build_image_embeddings(self):
		"""Generate image embedding in a limited text space

		Inputs:
			self.VGG_pred_data
			self.NIC_pred_data

		Output:
			self.image_embeddings
		"""
		self.VGG_local_pred_data = tf.placeholder(tf.float32, shape=[None, 49, 512], name="ims_vgg_local_pred_data")
		self.VGG_global_pred_data = tf.placeholder(tf.float32, shape=[None, 4096], name="ims_vgg_global_pred_data")
		self.NIC_pred_data = tf.placeholder(tf.float32, shape=[None, self.config.dim_NIC], name="ims_NIC_pred_data")

		# Map image space features into a limited text space.
		#self.input_vgg = tf.layers.batch_normalization(self.VGG_local_pred_data, training=self.phase)
		self.input_vgg = self.VGG_local_pred_data

		with tf.variable_scope("IncV4_embedding") as IncV4_scope:
			IncV4_embedding = tf.contrib.layers.fully_connected(
				inputs=self.VGG_global_pred_data,
				num_outputs=self.config.dim,
				activation_fn=None,
				weights_initializer=self.initializer,
				biases_initializer=None,
				scope=IncV4_scope)

			#IncV4_embedding = tf.nn.dropout(tf.tanh(tf.reshape(IncV4_embedding, [-1, 49, 512])), 0.5)
			self.IncV4_embedding = tf.nn.relu(tf.layers.batch_normalization(IncV4_embedding, training=self.phase, name="IncBN"))
			#self.IncV4_embedding = tf.nn.l2_normalize(self.IncV4_embedding, 1)

		with tf.variable_scope("NIC_embedding") as NIC_scope:
			NIC_embedding = tf.contrib.layers.fully_connected(
				inputs=self.NIC_pred_data,
				num_outputs=self.config.dim,
				activation_fn=None,
				weights_initializer=self.initializer,
				biases_initializer=None,
				scope=NIC_scope)
			self.NIC_embedding = tf.nn.relu(tf.layers.batch_normalization(NIC_embedding, training=self.phase, name="NicBN"))


		#self.image_embedding = tf.nn.l2_normalize(tf.layers.batch_normalization(
		#							tf.add(self.IncV4_embedding, self.NIC_embedding),
		#							training=self.phase, name="SumBN"), 1, name="image_embedding")


	def build_seq_embeddings(self):
		"""Generate text embeddings

		Inputs:
			self.ls_pred_data
			self.input_mask

		Output:
			self.text_embedding
		"""
		self.ls_pred_data = tf.placeholder(tf.int64, shape=[None, None], name="ls_pred_data")
		self.input_mask = tf.placeholder(tf.int64, shape=[None, None], name='mask')

		with tf.variable_scope("seq_embedding"):
			embedding_map = tf.get_variable(
				name="word_embedding",
				shape=[self.config.n_words, self.config.dim_word],
				initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
			seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.ls_pred_data)

		# BLSTM
		lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.dim, state_is_tuple=True)
		lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.dim, state_is_tuple=True)

		lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(
			lstm_cell_fw,
			output_keep_prob=self.keep_prob)
		lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(
			lstm_cell_bw,
			output_keep_prob=self.keep_prob)

		with tf.variable_scope("lstm") as lstm_scope:
			initial_state_fw = lstm_cell_fw.zero_state(batch_size=tf.shape(self.ls_pred_data)[0], dtype=tf.float32)
			initial_state_bw = lstm_cell_bw.zero_state(batch_size=tf.shape(self.ls_pred_data)[0], dtype=tf.float32)

			sequence_length = tf.reduce_sum(self.input_mask, 1)

			outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
		                                                     cell_bw=lstm_cell_bw,
		                                                     inputs=seq_embeddings,
		                                                     sequence_length=sequence_length,
		                                                     initial_state_fw=initial_state_fw,
		                                                     initial_state_bw=initial_state_bw,
		                                                     scope=lstm_scope)

			self.outputs = tf.add(outputs[0], outputs[1]) / 2
			self.final_state = tf.add(final_state[0][1], final_state[1][1]) / 2

			#self.text_embedding = tf.nn.l2_normalize(tf.add(final_state[0][1], final_state[1][1]) / 2, 1, name="text_embedding")

	def attention_layer(self):
		def fc_layer(bottom, in_size, out_size, name):
			with tf.variable_scope(name):
				weights = tf.get_variable(name=name+"_weights", shape=[in_size, out_size], initializer=self.initializer)
				biases = tf.get_variable(name=name+"_biases", shape=[out_size], initializer=self.initializer)

				x = tf.reshape(bottom, [-1, in_size])
				fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

				return fc

		def compute_init_memory():
			#u0 = tf.reduce_mean(self.outputs, 1)
			u0 = self.final_state

			#v_mean = tf.reduce_mean(self.input_vgg, 1)
			#with tf.variable_scope('P') as scope:
		        #       pv0_linear = fc_layer(v_mean, 512, self.config.dim, "P_0")
			#v0 = tf.nn.dropout(tf.nn.tanh(pv0_linear), self.keep_prob)

			#v0 = tf.layers.batch_normalization(tf.add(v0, self.NIC_embedding), training=self.phase)
			v0 = tf.layers.batch_normalization(tf.add(self.IncV4_embedding, self.NIC_embedding), training=self.phase)

			v0 = tf.nn.l2_normalize(v0, 1)
			u0 = tf.nn.l2_normalize(u0, 1)
			sim0 = tf.matmul(v0, u0, transpose_b=True)
			self.vfeats_all.append(v0)
			self.sfeats_all.append(u0)
			return v0, u0, sim0

		self.memories_v[0], self.memories_u[0], self.sim = compute_init_memory()

		#lstm_cell_v = tf.contrib.rnn.BasicLSTMCell(self.config.dim, state_is_tuple=True)
		#lstm_cell_v = tf.contrib.rnn.DropoutWrapper(lstm_cell_v, output_keep_prob=self.keep_prob)

		#lstm_cell_u = tf.contrib.rnn.BasicLSTMCell(self.config.dim, state_is_tuple=True)
		#lstm_cell_u = tf.contrib.rnn.DropoutWrapper(lstm_cell_u, output_keep_prob=self.keep_prob)

		for step in range(1,3):
			with tf.variable_scope("Wvm") as scope:
				Wvm_linear = fc_layer(self.memories_v[step-1], self.config.dim, 512, "Wvm_%d"%step)
				Wvm_linear = tf.reshape(Wvm_linear, [-1, 1, 512])
				Wvm_out = tf.nn.dropout(tf.nn.tanh(Wvm_linear), self.keep_prob)

			with tf.variable_scope("Wv") as scope:
				Wv_linear = fc_layer(self.input_vgg, self.config.dim_VGG, 512, "Wv_%d"%step)
				Wv_linear = tf.reshape(Wv_linear, [-1, 49, 512])
				Wv_out = tf.nn.dropout(tf.tanh(Wv_linear), self.keep_prob)

			hvn = tf.multiply(Wv_out, Wvm_out)

			with tf.variable_scope("Wvh") as scope:
				Wvh_linear = fc_layer(hvn, 512, 1, "Wvh_%d"%step)
				Wvh_linear = tf.reshape(Wvh_linear, [-1, 49])
				self.avn[step] = tf.nn.softmax(Wvh_linear)

			weighted_vfeats = tf.reduce_sum(tf.expand_dims(self.avn[step], 2)*self.input_vgg, 1)
			with tf.variable_scope("P") as scope:
				p_linear = fc_layer(weighted_vfeats, self.config.dim_VGG, self.config.dim, "P_%d"%step)
				#v = tf.nn.dropout(tf.tanh(p_linear), self.keep_prob)
				v = tf.nn.relu(tf.layers.batch_normalization(p_linear, training=self.phase, name="img_emb_%d"%step))

				v = tf.layers.batch_normalization(tf.add(v, self.NIC_embedding), training=self.phase, name="PV_%d"%step)

			with tf.variable_scope("Wum") as scope:
				Wum_linear = fc_layer(self.memories_u[step-1], self.config.dim, 512, "Wum_%d"%step)
				Wum_linear = tf.reshape(Wum_linear, [-1, 1, 512])
				Wum_out = tf.nn.dropout(tf.tanh(Wum_linear), self.keep_prob)

			with tf.variable_scope("Wu") as scope:
				Wu_linear = fc_layer(self.outputs, self.config.dim, 512, "Wu_%d"%step)
				Wu_linear = tf.reshape(Wu_linear, [tf.shape(self.ls_pred_data)[0], -1, 512])
				Wu_out = tf.nn.dropout(tf.nn.tanh(Wu_linear), self.keep_prob)

			hut = tf.multiply(Wu_out, Wum_out)

			with tf.variable_scope("Wuh") as scope:
				Wuh_linear = fc_layer(hut, 512, 1, "Wuh_%d"%step)
				Wuh_linear = tf.reshape(Wuh_linear, [tf.shape(self.ls_pred_data)[0], -1])
				self.aut[step] = tf.nn.softmax(Wuh_linear)

			u = tf.reduce_sum(tf.expand_dims(self.aut[step], 2)*self.outputs, 1)
			"""
			with tf.variable_scope('v_LSTM'):
				if step==1:
					state_v = lstm_cell_v.zero_state(batch_size=tf.shape(self.VGG_local_pred_data)[0], dtype=tf.float32)
					_, state_v = lstm_cell_v(self.memories_v[0], state_v)
				else:
					tf.get_variable_scope().reuse_variables()

				#v = tf.expand_dims(v, axis=1)
				v_cell_output, state_v = lstm_cell_v(v, state_v)
				self.memories_v[step] = v_cell_output

			with tf.variable_scope('u_LSTM'):
				if step==1:
					state_u = lstm_cell_u.zero_state(batch_size=tf.shape(self.ls_pred_data)[0], dtype=tf.float32)
					_, state_u = lstm_cell_u(self.memories_u[0], state_u)
				else:
					tf.get_variable_scope().reuse_variables()

				#v = tf.expand_dims(v, axis=1)
				u_cell_output, state_u = lstm_cell_u(u, state_u)
				self.memories_u[step] = u_cell_output
			"""

			v = tf.nn.l2_normalize(v, 1)
			u = tf.nn.l2_normalize(u, 1)
			self.memories_v[step] = tf.add(self.memories_v[step-1], v)
			self.memories_u[step] = tf.add(self.memories_u[step-1], u)

			self.sim = self.sim + tf.matmul(v, u, transpose_b=True)
			self.vfeats_all.append(v)
			self.sfeats_all.append(u)

		#self.ims_feature = tf.nn.l2_normalize(self.memories_v[2], 1)

		#self.ls_feature = tf.nn.l2_normalize(self.memories_u[2], 1)


	def build_loss(self):
		"""Builds the pairwise ranking loss function.

		Inputs:
			self.image_embedding
			self.text_embedding

		Output:
			self.embedding_loss
		"""

		with tf.name_scope("pairwise_ranking_loss"):
			# Compute losses
			#pred_score = tf.matmul(self.IncV4_embedding, self.text_embedding, transpose_b=True)
			pred_score = self.sim
			diagonal = tf.diag_part(pred_score)
			cost_s = tf.maximum(0., self.config.margin - diagonal + pred_score)
			cost_im = tf.maximum(0., self.config.margin - tf.reshape(diagonal, [-1, 1]) + pred_score)
			cost_s = tf.multiply(cost_s, (tf.ones([tf.shape(self.VGG_local_pred_data)[0], tf.shape(self.VGG_local_pred_data)[0]]) - tf.eye(tf.shape(self.VGG_local_pred_data)[0])))
			cost_im = tf.multiply(cost_im, (tf.ones([tf.shape(self.VGG_local_pred_data)[0], tf.shape(self.VGG_local_pred_data)[0]]) - tf.eye(tf.shape(self.VGG_local_pred_data)[0])))

                        #cost_s = tf.reduce_max(cost_s, 0)
                        #cost_im = tf.reduce_max(cost_im, 1)

			self.embedding_loss = tf.reduce_sum(cost_s) + tf.reduce_sum(cost_im)


	def build_optimizer(self):
		"""Initialize an optimizer.

		Inputs:
			All trainable variables

		Output:
			self.updates
		"""
		tvars = tf.trainable_variables()
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			grads, _ = tf.clip_by_global_norm(tf.gradients(self.embedding_loss, tvars), self.config.clip_gradients)
			optimizer = tf.train.AdamOptimizer(self.learning_rate)
			self.updates = optimizer.apply_gradients(zip(grads, tvars))


	def build(self):
		"""Create all ops for training."""
		self.build_seq_embeddings()
		self.build_image_embeddings()
		self.attention_layer()
		self.build_loss()
		self.build_optimizer()


