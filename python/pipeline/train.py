from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import datetime
from sklearn import metrics

from siamese.dataset import Dataset
from siamese.dataset import DataLoader
from siamese.model import *
from extensies import plotting
from extensies import augmentation as aug
from extensies import preprocessing as ps
from extensies import metrics as my_metrics 


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('train_iter', 2000, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_string('model', 'xmass', 'model to run')
flags.DEFINE_string('path_to_data', '../../data/', 'Path to data')


modalities = ['adc/t/20x20x1','cbval/t/20x20x1','ktrans/t/20x20x1']
augmentation = True

network_name,size = ps.generate_name(modalities)
df = pd.read_csv('records.csv',index_col=0)
row = pd.DataFrame(columns = df.columns)
row.loc[0,'iterations'] = FLAGS.train_iter
row.loc[0,'model'] = FLAGS.model
row.loc[0,'modality'] = modalities
row.loc[0,'size'] = size
row.loc[0,'date'] = datetime.datetime.now().strftime("%Y-%m-%d")

if __name__ == "__main__":


	# Here we load data and create train, test set with DataLoader 
	if len(modalities) > 1:
		loader = DataLoader(FLAGS.path_to_data,modalities)
		loader.load_data()
		loader.combine_channels(modalities)
		X_train, X_test, y_train, y_test = loader.get_train_test('combined')

	else:
		loader = DataLoader(FLAGS.path_to_data,modalities)
		loader.load_data()
		X_train, X_test, y_train, y_test = loader.get_train_test(modalities[0])
	
	row.loc[0,'normalization'] = 'ScaleNormalization'

	## LOCAL BINARY PATTERN
	# lbp = aug.LocalBinaryPattern(8,1,'uniform')
	# X_train = lbp.transform(X_train)
	# X_test = lbp.transform(X_test)

	# MNIST DATASET
	
	# (X_train,y_train),(X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
	# X_train = X_train.reshape(-1,28,28,1)
	# X_test = X_test.reshape(-1,28,28,1)
	# X_train = X_train / 255
	# X_test = X_test / 255

	
	## AUGMNETATION
	# create augmentor if we want deform out data
	if augmentation:
		augmentor = aug.ClassicAugmentor(X_train)
		row.loc[0,'augmentation'] = augmentor.name
	else:
		augmentor = False
		row.loc[0,'augmentation'] = False

	## DATASET
	# initialize Dataset for training process
	dataset = Dataset()
	dataset.images_train = X_train
	dataset.images_test = X_test
	dataset.labels_train = y_train
	dataset.labels_test = y_test


	row.loc[0,'num_of_data'] = len(dataset.images_train) + len(dataset.images_test)
	print('Shape of images: {},  Shape of labels: {}'.format(X_train.shape, y_train.shape))


	# initialize plots to visualize training
	plotter = plotting.DynamicPlot()
	plotter.on_launch(0,FLAGS.train_iter)

	plotter2 = plotting.DynamicPlot()
	plotter2.on_launch(0,FLAGS.train_iter)
	

	batch_train_losses = []
	batch_test_losses = []
	accuracy_train = []
	accuracy_test = []
	loss_train = []
	loss_test = []

	# we define our CNN model, we call function from siamese/model
	model = xmas_model
	row.loc[0,'name'] = 'xmas'

	# Setup network
	placeholder_shape = [None] + list(dataset.images_train.shape[1:])
	left = tf.placeholder(tf.float32, placeholder_shape, name='left')
	right = tf.placeholder(tf.float32, placeholder_shape, name='right')
	with tf.name_scope("similarity"):
	    label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	    label_float = tf.to_float(label)
	margin = 0.5
	left_output = model(left, reuse=False)
	right_output = model(right, reuse=True)
	loss = contrastive_loss( left_output, right_output,label_float, margin)

	next_batch = dataset.get_siamese_batch
	global_step = tf.Variable(0, trainable=False)

# 	starter_learning_rate = 0.0001
# 	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# 	# tf.scalar_summary('lr', learning_rate)
# 	train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

#   train_step = tf.train.MomentumOptimizer(0.0001, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

	train_step = tf.train.AdamOptimizer(0.00001).minimize(loss, global_step=global_step)


	# Start Training
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#setup tensorboard	
		tf.summary.scalar('step', global_step)
		tf.summary.scalar('loss', loss)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.op.name, var)
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('train.log', sess.graph)

		#train iter
		for i in range(FLAGS.train_iter):

			# Training data
			batch_left, batch_right, batch_similarity = next_batch(FLAGS.batch_size, augmentor = augmentor)
			_, l_train, summary_str = sess.run([train_step, loss, merged],
										 feed_dict={left:batch_left, right:batch_right, label: batch_similarity})
				
			# Test data
			batch_left, batch_right, batch_similarity = next_batch(FLAGS.batch_size,source = 'test')
			l_test= sess.run(loss,
										 feed_dict={left:batch_left, right:batch_right, label: batch_similarity})        
	        
			batch_train_losses.append(l_train)
			batch_test_losses.append(l_test)


			writer.add_summary(summary_str, i)
			print("\r#%d - Loss"%i,l_train,  "Test_Loss ", l_test)
				
			# here we visualize process for n iterations	
			if (i + 1) % FLAGS.step == 0:
				
				# run one shot test
				a = dataset.test_oneshot(sess,left_output,left,4,100)
				b = dataset.test_oneshot(sess,left_output,left,4,100,'train')
				print('Accuracy train: ',b)
				print('Accuracy test: ',a)
				accuracy_test.append(a)
				accuracy_train.append(b)

				loss_test.append(np.sum(batch_test_losses) / len(batch_test_losses))
				loss_train.append(np.sum(batch_train_losses) / len(batch_train_losses))


				train_feat = sess.run(left_output, feed_dict={left:dataset.images_train})
				search_feat = sess.run(left_output, feed_dict={left:dataset.images_test})
				y_pred,y_pred_t = [],[]
				for _,feat in enumerate(search_feat):
				    #calculate the cosine similarity and sort
					y_pred.append(my_metrics.siamese_predict(train_feat,feat,dataset))
					y_pred_t.append(my_metrics.treshold_predict(train_feat,feat,dataset,0.4,10))

				plotter.on_update(i+1,np.sum(batch_train_losses) / len(batch_train_losses),np.sum(batch_test_losses) / len(batch_test_losses))
				plotter2.on_update(i+1,metrics.roc_auc_score(dataset.labels_test,y_pred_t),metrics.roc_auc_score(dataset.labels_test,y_pred))
				

				batch_train_losses = []
				batch_test_losses = []
            				
			
		plotter.on_finish()	
		saver.save(sess, "model/"+network_name+".ckpt")

		df = df.append(row, ignore_index = True)
		df.to_csv('records.csv')
