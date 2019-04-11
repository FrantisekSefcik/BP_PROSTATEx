import tensorflow as tf
import pandas as pd
from siamese.dataset import Dataset
from siamese.model import *
from extensies import metrics as my_metrics

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
import ast



def evaluate_model(dataset,img_placeholder, net, model_name):
	df = pd.read_csv('records.csv',index_col=0)

	# restore model and count features
	# img_placeholder = tf.placeholder(tf.float32, [None] + list(dataset.images_train.shape[1:]), name='img')

	#run the train image through the network to get the test features
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    ckpt = tf.train.get_checkpoint_state("model")
	    saver.restore(sess, "model/" + model_name + ".ckpt")
	    train_feat = sess.run(net, feed_dict={img_placeholder:dataset.images_train})

	#run the test image through the network to get the test features
	saver = tf.train.Saver()
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    ckpt = tf.train.get_checkpoint_state("model")
	    saver.restore(sess, "model/" + model_name + ".ckpt")
	    search_feat = sess.run(net, feed_dict={img_placeholder:dataset.images_test})


	y_pred = []
	y_pred_t = []
	y_pred_w = []
	for _,feat in enumerate(search_feat):
	    #calculate the cosine similarity and sort
	    y_pred.append(my_metrics.siamese_predict(train_feat,feat,dataset))
	    y_pred_t.append(my_metrics.treshold_predict(train_feat,feat,dataset,0.4,10))
	    y_pred_w.append(my_metrics.weighted_predict(train_feat,feat,dataset,0.4,10))


	acc1 = accuracy_score(dataset.labels_test,y_pred)
	auc1 = metrics.roc_auc_score(dataset.labels_test,y_pred)

	acc2 = accuracy_score(dataset.labels_test,y_pred_t)
	auc2 = metrics.roc_auc_score(dataset.labels_test,y_pred_t)

	acc3 = accuracy_score(dataset.labels_test,y_pred_w)
	auc3 = metrics.roc_auc_score(dataset.labels_test,y_pred_w)

	idx = df[df['name'] == model_name].index
	df.loc[idx,'acc'] = acc1
	df.loc[idx,'auc'] = auc1
	df.loc[idx,'acc_p'] = acc2
	df.loc[idx,'auc_p'] = auc2
	df.loc[idx,'acc_w'] = acc3
	df.loc[idx,'auc_w'] = auc3
	
	print("Model: {0}: Acc: {1:.4f}, {2:.4f},  AUC: {3:.4f}, {4:.4f}".format(model_name,acc1, acc2, auc1, auc2))


	df.to_csv('records.csv')

	return acc1,auc1,acc2,auc2




def evaluate_kfold(index):

	df = pd.read_csv('records.csv',index_col=0) 


	for idx,row in df[df['kfold_id'] == index].iterrows():

		model_name = row['name']
		modalities = row['modality']
		modalities = ast.literal_eval(modalities)

		if len(modalities)>1:
		    print('combined')
		    loader = DataLoader('../../data/',modalities)
		    loader.load_data()
		    X,y = loader.combine_channels(modalities)
		    X_train, X_test, y_train, y_test = loader.get_train_test('combined')

		else:
		    loader = DataLoader('../../data/',modalities)
		    loader.load_data()
		    X_train, X_test, y_train, y_test = loader.get_train_test(modalities[0])

		dataset = Dataset()
		dataset.images_train = X_train
		dataset.images_test = X_test
		dataset.labels_train = y_train
		dataset.labels_test = y_test


		# restore model and count features
		img_placeholder = tf.placeholder(tf.float32, [None] + list(dataset.images_train.shape[1:]), name='img')
		net = xmas_model(img_placeholder, reuse=False)

		#run the train image through the network to get the test features
		saver = tf.train.Saver()
		with tf.Session() as sess:
		    sess.run(tf.global_variables_initializer())
		    ckpt = tf.train.get_checkpoint_state("model")
		    saver.restore(sess, "model/" + model_name + ".ckpt")
		    train_feat = sess.run(net, feed_dict={img_placeholder:dataset.images_train})

		#run the test image through the network to get the test features
		saver = tf.train.Saver()
		with tf.Session() as sess:
		    sess.run(tf.global_variables_initializer())
		    ckpt = tf.train.get_checkpoint_state("model")
		    saver.restore(sess, "model/" + model_name + ".ckpt")
		    search_feat = sess.run(net, feed_dict={img_placeholder:dataset.images_test})
		    



		y_pred = []
		y_pred_t = []
		y_pred_w = []
		for _,feat in enumerate(search_feat):
		    #calculate the cosine similarity and sort
		    y_pred.append(my_metrics.siamese_predict(train_feat,feat,dataset))
		    y_pred_t.append(my_metrics.treshold_predict(train_feat,feat,dataset,0.4,10))
		    y_pred_w.append(my_metrics.weighted_predict(train_feat,feat,dataset,0.4,10))


		acc = accuracy_score(dataset.labels_test,y_pred)
		auc = metrics.roc_auc_score(dataset.labels_test,y_pred)

		df.loc[idx,'acc'] = acc
		df.loc[idx,'auc'] = auc

		print("First choice - Accuracy: {}, AUC: {}".format( acc, auc))

		acc = accuracy_score(dataset.labels_test,y_pred_t)
		auc = metrics.roc_auc_score(dataset.labels_test,y_pred_t)

		df.loc[idx,'acc_p'] = acc
		df.loc[idx,'auc_p'] = auc
		print("Percentage -   Accuracy: {}, AUC: {}".format(acc, auc))

		acc = accuracy_score(dataset.labels_test,y_pred_w)
		auc = metrics.roc_auc_score(dataset.labels_test,y_pred_w)

		df.loc[idx,'acc_w'] = acc
		df.loc[idx,'auc_w'] = auc

		print("Weighted -     Accuracy: {}, AUC: {}".format(acc, auc))

		df.to_csv('records.csv')