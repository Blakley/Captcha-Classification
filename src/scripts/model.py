import os

# suppress tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Captcha Classification Model
class Captcha():
	def __init__(self):
		pass


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

		# define RGB image sizes
		self.shape = (128, 128, 3)
		
		# create model next
		self.create()

	'''
		================================
		
		================================
	'''	
	def create(self):
		# create sequential model
		self._model = models.Sequential()

		# add input layer
		self._model.add(layers.Input(shape = self.shape))

		# add convolutional layers
		self._model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
		self._model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		
		self._model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
		self._model.add(layers.MaxPooling2D(pool_size=(2, 2)))
		
		self._model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
		self._model.add(layers.MaxPooling2D(pool_size=(2, 2)))

		# flatten layer
		self._model.add(layers.Flatten())

		# add dense layers : softmax activation for multi-class classification
		self._model.add(layers.Dense(128, activation='relu'))
		self._model.add(layers.Dense(len(self.classes), activation='softmax'))

		# compile model next
		self.compile()

	'''
		================================
		
		================================
	'''	
	def compile(self):
		model.compile(
			optimizer='adam', 
			loss='sparse_categorical_crossentropy', 
			metrics=['accuracy']
		)

		# train model next
		self.train()

	'''
		================================
		
		================================
	'''	
	def train(self):
		self.history = model.fit(
			self.train, 
			validation_data = self.validation, 
			epochs = 10
		)


	'''
		================================
		
		================================
	'''	
	def statistics(self):
		pass

	'''
		================================
		
		================================
	'''	
	def save(self):
		pass

	'''
		================================
		
		================================
	'''	
	def load(self):
		pass


# 
if __name__ == '__main__':
	# model instance
	_model = Captcha()

	# model setup outline
	_model.setup()

	# 
