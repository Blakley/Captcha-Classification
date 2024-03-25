import os

# suppress tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Captcha Classification Model
class Captcha():
	def __init__(self):
		# 
		self.setup()


	'''
		================================

		================================
	'''	
	def setup(self):
		# load dataset & split dataset
		self.train, self.validation = keras.utils.image_dataset_from_directory (
			"../dataset",
		    subset = "both",
		    seed = 8888,
		    shuffle = True,
		    batch_size = 32,
		    image_size = (128, 128),
		    validation_split = 0.2,
		)

		self.classes = self.train.class_names
		# print(self.classes)


	'''
		================================
		
		================================
	'''	
	def function(self):
		pass


	'''
		================================
		
		================================
	'''	
	def function(self):
		pass


	'''
		================================
		
		================================
	'''	
	def function(self):
		pass

# 
if __name__ == '__main__':
	# model instance
	_model = Captcha()