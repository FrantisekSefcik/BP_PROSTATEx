import tensorflow as tf
import pandas as pd
from siamese.dataset import Dataset
from siamese.dataset import DataLoader
from siamese.model import *
from extensies import metrics as my_metrics

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report
from extensies import augmentation as aug
import ast

df = pd.read_csv('records.csv',index_col=0)
idx = df.tail(1).index[0] 

model_name = df.iloc[idx]['name']
modalities = df.iloc[idx]['modality']
modalities = ast.literal_eval(modalities)
combined = len(modalities)>1

if combined:
	loader = DataLoader('../../data/',modalities)
	loader.load_data()
	X,y = loader.combine_channels(modalities)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # X_train, X_test, y_train, y_test = loader.get_train_test('combined')

else:
	loader = DataLoader('../../data/',modalities)
	loader.load_data()
	X_train, X_test, y_train, y_test = loader.get_train_test(modalities[0])

print('Model name: ' ,model_name)


# lbp = aug.LocalBinaryPattern(8,1,'uniform')
# X_train = lbp.transform(X_train)
# X_test = lbp.transform(X_test)

# (X_train,y_train),(X_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
# X_train = X_train.reshape(-1,28,28,1)
# X_test = X_test.reshape(-1,28,28,1)
# X_train = X_train / 255
# X_test = X_test / 255

dataset = Dataset()
dataset.images_train = X_train
dataset.images_test = X_test
dataset.labels_train = y_train
dataset.labels_test = y_test


# restore model and count features
img_placeholder = tf.placeholder(tf.float32, [None] + list(dataset.images_train.shape[1:]), name='img')
net = mnist_model(img_placeholder, reuse=False)

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