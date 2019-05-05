from imgaug import augmenters as iaa
import random
import numpy as np 
import skimage.feature as ft
import albumentations



class ElasticAugmentor(object):

	def __init__(self,images):
		w,h,_ = images[0].shape
		self.seq = iaa.Sequential([
		    
		    # iaa.ElasticTransformation(sigma=5.0,alpha = 20,cval = 0),
		    iaa.PiecewiseAffine(scale = (0.01,0.05),mode="symmetric"),
		    iaa.Fliplr(0.5), # horizontally flip 50% of the images
		    iaa.Flipud(0.5),
		    
		    iaa.Rot90(k=(0,3)),

		    # iaa.CropToFixedSize(w - 2,w - 2,position = 'uniform') 
		])
		self.name = 'Elastic'

	def generate_images(self,images,labels,num):
		lenght = len(images)

		X_arr = []
		y_arr = []

		for x in range(num):
			random_image_index = random.randint(0, lenght -1)
			img = self.seq.augment_image(images[random_image_index])
			X_arr.append(img)
			y_arr.append(labels[random_image_index])

		return np.array(X_arr),np.array(y_arr)



class RigidAugmentor(object):

	def __init__(self,images):
		w,h,_ = images[0].shape
		self.seq = iaa.Sequential([
		    # iaa.Sometimes(0.5,
		       # iaa.ElasticTransformation(sigma=5.0,alpha = 20)),
		#     iaa.PiecewiseAffine(scale = (0,0.05)),
		    iaa.Fliplr(0.5), # horizontally flip 50% of the images
		    iaa.Flipud(0.5), 
		    iaa.Rot90(k=(0,3))

		    # iaa.CropToFixedSize(w - 2,w - 2,position = 'uniform') 
		])

		self.name = 'Rigid'

	def generate_images(self,images,labels,num):
		lenght = len(images)

		X_arr = []
		y_arr = []

		return self.seq.augment_images(images),labels
		# for x in range(num):
		# 	random_image_index = random.randint(0, lenght -1)
		# 	img = self.seq.augment_image(images[random_image_index])
		# 	X_arr.append(img)
		# 	y_arr.append(labels[random_image_index])

		# return np.array(X_arr),np.array(y_arr)

# Class for loval binary pathern
class LocalBinaryPattern(object):

	def __init__(self,P=8,R=1,method = 'uniform'):
		self.R = R
		self.P = P
		self.METHOD = method


	def transform(self,images):
		shape = images.shape
		images = [ft.local_binary_pattern(img.reshape((shape[1],shape[2])), 
												self.P, self.R, self.METHOD) for img in images]


		return np.array(images).reshape(shape)

# Rigid deformation with horizontal flip, rotation, shifting and scaling
class ClassicAugmentor(object):

	def __init__(self,images):
		self.seq= albumentations.Compose([
			        albumentations.HorizontalFlip(p=0.5),
			        albumentations.RandomRotate90(p = 1),
			        albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=20, p=0.3)
			     ], p=1)

		self.name = 'Rigid'
	# generate numbers of images from original images 
	def generate_images(self,images,labels,num):
		lenght = len(images)

		X_arr = []
		y_arr = []

		for x in range(num):
			random_image_index = random.randint(0, lenght -1)
			img = self.seq(image = images[random_image_index])['image']
			X_arr.append(img)
			y_arr.append(labels[random_image_index])

		return np.array(X_arr),np.array(y_arr)
	# function to use in Dataset object method get_siamese_batch
	def augment_image(self,image):

		return self.seq(image = image)['image']

# Elastic deformation + fliping and rotation
class Elastic2Augmentor(object):

	def __init__(self,images):
		self.seq= albumentations.Compose([
				    albumentations.augmentations.transforms.ElasticTransform(alpha=1, sigma=20, alpha_affine=3, 
				                                                             interpolation=1, border_mode=4, 
				                                                             always_apply=False, approximate=False, p=1),
				    albumentations.HorizontalFlip(p=0.5),
			        albumentations.RandomRotate90(p = 1)
				    ], p=1)

		self.name = 'Elastic'

	def generate_images(self,images,labels,num):
		lenght = len(images)

		X_arr = []
		y_arr = []

		for x in range(num):
			random_image_index = random.randint(0, lenght -1)
			img = self.seq(image = images[random_image_index])['image']
			X_arr.append(img)
			y_arr.append(labels[random_image_index])

		return np.array(X_arr),np.array(y_arr)
	# function to use in Dataset object method get_siamese_batch
	def augment_image(self,image):

		return self.seq(image = image)['image']

