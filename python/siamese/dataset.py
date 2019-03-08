from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals
from scipy.spatial.distance import cdist

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import SimpleITK as sitk
from functools import reduce

class Dataset(object):
    images_train = np.array([])
    images_test = np.array([])
    labels_train = np.array([])
    labels_test = np.array([])
    unique_label = np.array([])
    map_label_indices = dict()
    updated = False

    def _get_siamese_similar_pair(self, source):
        label =np.random.choice(self.unique_label)
        l, r = np.random.choice(self.map_label_indices[source][label], 2, replace=False)
        return l, r, 1, label

    def _get_siamese_dissimilar_pair(self, source):
        label_l, label_r = np.random.choice(self.unique_label, 2, replace=False)
        l = np.random.choice(self.map_label_indices[source][label_l])
        r = np.random.choice(self.map_label_indices[source][label_r])
        return l, r, 0, label_l
    
    def _get_dissimilar_image(self, source, label = 'auto', n = 1):
        if label == 'auto':
            target_label = np.random.choice(self.unique_label)
        else:
            target_label = np.random.choice(np.setdiff1d(self.unique_label,[label]))
        
        indecies = np.random.choice(self.map_label_indices[source][target_label],(n,))
        return indecies, np.zeros(n)
        

    def _get_siamese_pair(self,source):
        if np.random.random() < 0.5:
            return self._get_siamese_similar_pair(source)
        else:
            return self._get_siamese_dissimilar_pair(source)

    def get_siamese_batch(self, n, source = 'train'):
        
        if not self.updated:
            self._update()
        
        idxs_left, idxs_right, labels = [], [], []
        for _ in range(n):
            l, r, x, _ = self._get_siamese_pair(source)
            idxs_left.append(l)
            idxs_right.append(r)
            labels.append(x)
        
        images = self._get_images(source)
            
        return images[idxs_left,:], images[idxs_right, :], np.expand_dims(labels, axis=1)
    
    
    def get_oneshot_task(self, n = 2, source = 'test'):
        
        #find one similar pair
        l, r, x, label = self._get_siamese_similar_pair(source)
        # rest of pairs will by from different classies
        r_negativ, zeros = self._get_dissimilar_image(source, label = label, n = (n-1) )
        
        idxs_left = np.full((n,), l)
        idxs_right = np.concatenate(([r], r_negativ))
        
        images = self._get_images(source)
        labels = np.append([x], zeros)
        return images[idxs_left,:], images[idxs_right, :], labels
    
    def test_oneshot(self,sess,net,placeholder,N,k,source = 'test'):
        
        if not self.updated:
            self._update()
        
        n_correct = 0
        
        for i in range(k):
            left,right, targets = self.get_oneshot_task(N,source)
            right_feat = sess.run(net, feed_dict={placeholder: right})
            original_feat = sess.run(net, feed_dict={placeholder:left})
            
            #calculate the cosine similarity and sort
            dist = cdist(right_feat, [original_feat[0]], 'cosine')
            rank = np.argsort(dist.ravel())
            
            
            if rank[0] == 0:
                n_correct+=1
                
        percent_correct = (100.0 * n_correct / k)
        
        return percent_correct
        
    
    def _get_images(self,source = 'train'):
        if source == 'test':
            images = self.images_test
        else:
            images = self.images_train
        return images
    
    def _update(self):
        
        if self.images_train.size == 0:
            raise NameError('Empty Train Array !')
            
        self.unique_label = np.unique(self.labels_train)
        self.map_label_indices = {'train' : {label: np.flatnonzero(self.labels_train == label) for label in self.unique_label},
                                  'test' : {label: np.flatnonzero(self.labels_test == label) for label in self.unique_label}}
        self.updated = True
        
        
class DataLoader(object):
    
    def __init__(self,path_to_data = '.', subdirs_paths = []):
        self.path_to_data = path_to_data
        self.subdirs = set(subdirs_paths)
        self.images_array  = {}
        self.labels_array  = {}
        self.df_info = {}
        
    def load_data(self,subdirs_paths = None):
        
        if subdirs_paths:
            self.subdirs = self.subdirs.union(subdirs_paths)
        
        for path in self.subdirs:
            # open dataframe with info about dataset
            self.df_info[path] = pd.read_csv(os.path.join(self.path_to_data,path,'info.csv'))
            # read images
            self.images_array[path]  = np.array([sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.path_to_data,path,name))) for name  in self.df_info[path]['name']])
            # reshape images
            shape = self.images_array[path].shape
            self.images_array[path] = self.images_array[path].reshape(-1,shape[1],shape[2],1)
            # load labels
            self.labels_array[path]  = np.array([int(x) for x in self.df_info[path]['ClinSig']])
        
        return self.images_array
    
    def get_data(self,subdir):
        
        return self.images_array[subdir], self.labels_array[subdir]
    
    def combine_channels(self,subdirs_to_combine):
        
        if any(sub not in self.subdirs for sub in subdirs_to_combine):
            raise Exception('Paths have not been leaded!')
        
        if not reduce(lambda x,y: y 
                      if len(self.images_array[x]) == len(self.images_array[y]) else False , subdirs_to_combine):
            raise Exception('Not same number of images for all paths!')
            
        if not reduce(lambda x,y: y 
                      if self.labels_array[x] == self.images_array[y] else False , subdirs_to_combine):
            raise Exception('Not identicall labels to concatenate!')
            
        images = [self.images_array[sub] for sub in subdirs_to_combine]
        concated_image = np.concatenate(images, axis=3)
        
        return concated_image

        
    