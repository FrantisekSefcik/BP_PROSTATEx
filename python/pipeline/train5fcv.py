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
from extensies import evaluation
from extensies import augmentation as aug
from extensies import preprocessing as ps
from extensies import metrics as my_metrics
from functools import reduce 




flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('train_iter', 200, 'Total training iter')
flags.DEFINE_integer('step', 50, 'Save after ... iteration')
flags.DEFINE_string('model', 'xmass', 'model to run')
flags.DEFINE_string('path_to_data', '../../data/', 'Path to data')

modalities = ['adc/a/20x20x1','t2tsetra/a/20x20x1']
augmentation = False




if __name__ == "__main__":

	if len(modalities) > 1:
		loader = DataLoader(FLAGS.path_to_data,modalities)
		loader.load_data()
		loader.combine_channels(modalities)
		kfold = loader.k_fold(subdir = 'combined',num = 3)
		data_shape = loader.get_shape('combined')


	else:
		loader = DataLoader(FLAGS.path_to_data,modalities)
		loader.load_data()
		kfold = loader.k_fold(subdir = modalities[0],num = 3)
		data_shape = loader.get_shape(modalities[0])
	

	# Setup network
	epoch_accuracy = {}
	kfold_id = ps.generate_index()
	model = xmas_model
	placeholder_shape = [None] + list(data_shape[1:])

	left = tf.placeholder(tf.float32, placeholder_shape, name='left')
	right = tf.placeholder(tf.float32, placeholder_shape, name='right')
	with tf.name_scope("similarity"):
	    label = tf.placeholder(tf.int32, [None, 1], name='label') # 1 if same, 0 if different
	    label_float = tf.to_float(label)
	margin = 0.5
	left_output = model(left, reuse=False)
	right_output = model(right, reuse=True)
	loss = contrastive_loss( left_output, right_output,label_float, margin)
	# -- Setup network --


	for X_train,X_test,y_train,y_test in kfold:

		network_name,size = ps.generate_name(modalities)
		df = pd.read_csv('records.csv',index_col=0)
		row = pd.DataFrame(columns = df.columns)
		row.loc[0,'iterations'] = FLAGS.train_iter
		row.loc[0,'model'] = FLAGS.model
		row.loc[0,'modality'] = modalities
		row.loc[0,'size'] = size
		row.loc[0,'name'] = network_name
		row.loc[0,'date'] = datetime.datetime.now().strftime("%Y-%m-%d")
		row.loc[0,'normalization'] = 'ScaleNormalization'
		row.loc[0,'kfold_id'] = kfold_id

		dataset = Dataset()
		dataset.images_train = X_train
		dataset.images_test = X_test
		dataset.labels_train = y_train
		dataset.labels_test = y_test

		row.loc[0,'num_of_data'] = len(dataset.images_train) + len(dataset.images_test)
		print('Shape of images: {},  Shape of labels: {}'.format(X_train.shape, y_train.shape))


		## AUGMNETATION
		if augmentation:
			augmentor = aug.ClassicAugmentor(X_train)
			row.loc[0,'augmentation'] = augmentor.name
		else:
			augmentor = False
			row.loc[0,'augmentation'] = False

		next_batch = dataset.get_siamese_batch

		plotter = plotting.DynamicPlot()
		plotter.on_launch(0,FLAGS.train_iter)

		plotter2 = plotting.DynamicPlot()
		plotter2.on_launch(0,FLAGS.train_iter)
		

		batch_train_losses = batch_test_losses = []
		accuracy_train = accuracy_test = []
		loss_train = loss_test = []

		
		global_step = tf.Variable(0, trainable=False)
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
				batch_left, batch_right, batch_similarity = next_batch(FLAGS.batch_size,augmentor = augmentor)

				_, l_train, summary_str = sess.run([train_step, loss, merged],
											 feed_dict={left:batch_left, right:batch_right, label: batch_similarity})
					
				batch_left, batch_right, batch_similarity = next_batch(FLAGS.batch_size,source = 'test')
				l_test= sess.run(loss,
											 feed_dict={left:batch_left, right:batch_right, label: batch_similarity})        
		        
				batch_train_losses.append(l_train)
				batch_test_losses.append(l_test)


				writer.add_summary(summary_str, i)
				print("\r#%d - Loss"%i,l_train,  "Test_Loss ", l_test)
					
				if (i + 1) % FLAGS.step == 0:
					#generate test
					# TODO: create a test file and run per batch

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

			saver.save(sess, "model/"+network_name+".ckpt")

		df = df.append(row, ignore_index = True)
		df.to_csv('records.csv')
		epoch_accuracy[network_name] =  evaluation.evaluate_model(dataset, left, left_output, network_name)

	print("REPORT :")	
	for key, value in epoch_accuracy.items():
		print("Model: {0}: Acc: {1:.4f}, {2:.4f},  AUC: {3:.4f}, {4:.4f}".format(key, value[0], value[2], value[1], value[3]))

	avg_auc1 = reduce(lambda x, value:x + value[1], epoch_accuracy.values(), 0) / len(epoch_accuracy)
	avg_auc2 = reduce(lambda x, value:x + value[3], epoch_accuracy.values(), 0) / len(epoch_accuracy)
	avg_acc1 = reduce(lambda x, value:x + value[0], epoch_accuracy.values(), 0) / len(epoch_accuracy)
	avg_acc2 = reduce(lambda x, value:x + value[2], epoch_accuracy.values(), 0) / len(epoch_accuracy)
	print("Avg:                      Acc: {0:.4f}, {1:.4f},  AUC: {2:.4f}, {3:.4f}".format(avg_acc1, avg_acc2, avg_auc1, avg_auc2))