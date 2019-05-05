from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals
from scipy.spatial.distance import cdist

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import SimpleITK as sitk
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class Dataset(object):
    """ class Dataset represent class to store test and train 
    data for training. Also have method 'get_siamese_batch' that 
    return N images pairs with label of similarity.
    Method 'test_oneshot', creates one_shot scenario for train model
    """
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

    def get_siamese_batch(self, n, source = 'train',augmentor = False):
        """
        Args:
        n(int): number of images pairs in batch
        source(str): source of data, eather train, test or valid
        augmentor(Augmentation Object): class of augmentor defined at extensies/augmentation
        
        Return:
        np.arrays of images and labels
        """
        
        if not self.updated:
            self._update()
        
        images = self._get_images(source)
        images_left, images_right, labels = [], [], []
        
        for _ in range(n):
            l, r, x, _ = self._get_siamese_pair(source)
            if augmentor:
                images_left.append(augmentor.augment_image(images[l]))
                images_right.append(augmentor.augment_image(images[r]))
            else:
                images_left.append(images[l])
                images_right.append(images[r])
            labels.append(x)
        
            
        return np.array(images_left), np.array(images_right), np.expand_dims(labels, axis=1)
    
    
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
        
    def get_batch(self, n ,augmentor = False):
        """
        Args:
        n(int): number of images pairs in batch
        source(str): source of data, eather train, test or valid
        augmentor(Augmentation Object): class of augmentor defined at extensies/augmentation
        
        Return:
        np.arrays of images and labels
        """
        
        if not self.updated:
            self._update()
        
        images, labels = [],[]
        
        indecies = np.random.choice(len(self.images_train), n)
        
        for idx in indecies:
            if augmentor:
                images.append(augmentor.augment_image(self.images_train[idx]))
            else:
                images.append(self.images_train[idx])
            labels.append(self.labels_train[idx])
        
            
        return np.array(images), np.array(labels)
        
        
        
class DataLoader(object):
    
    """ class DataLoder represent class to read all images in different modalities
    from folder.
    Method 'get_train_test' return test and train set from loaded data.
    Method 'combine_channels' can combine sets of images to more channels images.
    Method 'get_kfold' return generator for kfold cross validation
    
    __init__():
    Args:
    path_to_data(str): root path for data
    subdirs_paths(list): name of folders from read images
    """
    
    def __init__(self,path_to_data = '.', subdirs_paths = []):
        self.path_to_data = path_to_data
        self.subdirs = set(subdirs_paths)
        self.images_array  = {}
        self.labels_array  = {}
        self.df_info = {}

    def get_normalization(self,subdir):

        return self.df_info[subdir].iloc[0]['normalization']
        
    def load_data(self,subdirs_paths = None):
        
        """
        Args:
        subdirs_paths(list): name of folders to read images from them
        """
        
        if subdirs_paths:
            self.subdirs = self.subdirs.union(subdirs_paths)
        
        for path in self.subdirs:
            # open dataframe with info about dataset
            self.df_info[path] = pd.read_csv(os.path.join(self.path_to_data,path,'info.csv'))
            # read images
            self.images_array[path]  = np.array([sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.path_to_data,path,name), sitk.sitkFloat32)) for name  in self.df_info[path]['name']])
            # reshape images
            shape = self.images_array[path].shape
            shape = shape[::-1]
            if len(shape) > 3:
                self.images_array[path] = self.images_array[path].reshape(-1,shape[0],shape[1],shape[2],1)
            else:
                self.images_array[path] = self.images_array[path].reshape(-1,shape[0],shape[1],1)

            
            # load labels
            self.labels_array[path]  = np.array([int(x) for x in self.df_info[path]['ClinSig']])
        
        return self.images_array
    
    def get_data(self,subdir):
        
        return self.images_array[subdir], self.labels_array[subdir]

    def get_shape(self,subdir):
        
        return self.images_array[subdir].shape

    def get_data_by_index(self,subdir = None, idx = 0):
        if subdir == None:
            images = []
            labels = []
            for subdir in self.subdirs:
                images.append(self.images_array[subdir][idx])
                labels.append(self.labels_array[subdir][idx])
            return np.array(images),np.array(labels)
        else:
            return self.images_array[subdir][idx], self.labels_array[subdir][idx]

    def get_train_test(self,subdir,test_size = 0.3,zones = ['PZ','TZ','AS']):
        
        """
        Args:
        subdir(str): name of folder with images
        test_size(float): percentage of test data 
        zones(list): zones of prostate
        
        Return:
        np.array of train_images, test_images, train_labels, test_labels
        """
        
        if(subdir == 'combined'):
            df = next(iter(self.df_info.values()))
        else:
            df = self.df_info[subdir]
            
        df = df[df['zone'].isin(zones)]

        idx_train, idx_test = train_test_split(df['ID'].unique(), test_size=test_size, random_state=42)

        idx_train  = df[df['ID'].isin(idx_train)].index
        idx_test  = df[df['ID'].isin(idx_test)].index
        return self.images_array[subdir][idx_train], self.images_array[subdir][idx_test], self.labels_array[subdir][idx_train], self.labels_array[subdir][idx_test]
    
    def combine_channels(self,subdirs_to_combine):
        
        """
        
        Args:
        subdirs(list of str): name of folders with images to combine
        
        Combined images are stored as 'combined', in next calls of 
        k_fold and get_train_test methods set argument subdir as combined 
        
        Return:
        np.array of train_images, test_images, train_labels, test_labels
        """
        
        if any(sub not in self.subdirs for sub in subdirs_to_combine):
            raise Exception('Paths have not been leaded!')
        
        if not reduce(lambda x,y: y 
                      if len(self.images_array[x]) == len(self.images_array[y]) else False , subdirs_to_combine):
            raise Exception('Not same number of images for all paths!')
            
        # if not reduce(lambda x,y: y 
        #               if self.labels_array[x] == self.images_array[y] else False , subdirs_to_combine):
        #     raise Exception('Not identicall labels to concatenate!')
            
        images = [self.images_array[sub] for sub in subdirs_to_combine]
        concated_image = np.concatenate(images, axis=3)
        
        self.images_array['combined'] = concated_image
        self.labels_array['combined'] = self.labels_array[subdirs_to_combine[0]]
        return concated_image, self.labels_array[subdirs_to_combine[0]]

    def k_fold(self,subdir,num=5,zones = ['PZ','TZ','AS']):
        
        """
        Args:
        subdir(str): name of folder with images
        num(int): number of folds
        zones(list): zones of prostate
        
        Return:
        np.array of train_images, test_images, train_labels, test_labels
        """

        if(subdir == 'combined'):
            df = next(iter(self.df_info.values()))
        else:
            df = self.df_info[subdir]
            
        df = df[df['zone'].isin(zones)]
        indices = df['ID'].unique()

        kf = KFold(n_splits=num,random_state=42)
        generator = kf.split(indices)

        for x,y in generator:
            idx_train  = df[df['ID'].isin(indices[x])].index
            idx_test  = df[df['ID'].isin(indices[y])].index
            yield self.images_array[subdir][idx_train], self.images_array[subdir][idx_test], self.labels_array[subdir][idx_train], self.labels_array[subdir][idx_test]
    
             





        
    