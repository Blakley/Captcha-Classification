import os

# suppress tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

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
		self.training, self.validation = keras.utils.image_dataset_from_directory (
			"../dataset",
		    subset = "both",
		    seed = 8888,
		    shuffle = True,
		    batch_size = 32,
		    image_size = (128, 128),
		    validation_split = 0.2,
		)

		self.classes = self.training.class_names
		# print(self.classes)

		# define RGB image sizes
		self.shape = (128, 128, 3)
		

	'''
		================================
		
		================================
	'''	
	def create(self):
	    # create sequential model
	    self._model = models.Sequential()

	    # add input layer
	    self._model.add(layers.Input(shape=self.shape))

	    # add convolutional layers
	    self._model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
	    self._model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	    self._model.add(layers.Dropout(0.2))  # Add dropout layer with dropout rate of 0.2

	    self._model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
	    self._model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	    self._model.add(layers.Dropout(0.2))  # Add dropout layer with dropout rate of 0.2

	    self._model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
	    self._model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	    self._model.add(layers.Dropout(0.2))  # Add dropout layer with dropout rate of 0.2

	    # flatten layer
	    self._model.add(layers.Flatten())

	    # add dense layers: softmax activation for multi-class classification
	    self._model.add(layers.Dense(128, activation='relu'))
	    self._model.add(layers.Dropout(0.4)) # Add dropout layer with dropout rate of 0.4
	    self._model.add(layers.Dense(len(self.classes), activation='softmax'))

	    # compile model next
	    self.compile()


	'''
		================================
		
		================================
	'''	
	def compile(self):
		self._model.compile(
			optimizer='adam', 
			loss='sparse_categorical_crossentropy', 
			metrics=['accuracy']
		)

		# view layers
		print(self._model.summary())
		
		# train model next
		self.train()

	'''
		================================
		
		================================
	'''	
	def train(self):
		self.history = self._model.fit(
			self.training, 
			validation_data = self.validation, 
			epochs = 10
		)

		# 
		self.statistics()


	'''
		================================
		
		================================
	'''	
	def statistics(self):
		# accuracy values recording during each epoch
		accuracy = self.history.history['accuracy']
		validation_accuracy = self.history.history['val_accuracy']

		loss = self.history.history['loss']
		validation_loss = self.history.history['val_loss']

		# visual accuracy and loss during each epoch
		epochs_range = range(10)

		plt.figure(figsize=(8, 8))
		plt.subplot(1, 2, 1)
		plt.plot(epochs_range, accuracy, label='Training Accuracy')
		plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
		plt.legend(loc='lower right')
		plt.title('Training and Validation Accuracy')
	
		plt.subplot(1, 2, 2)
		plt.plot(epochs_range, loss, label='Training Loss')
		plt.plot(epochs_range, validation_loss, label='Validation Loss')
		plt.legend(loc='upper right')
		plt.title('Training and Validation Loss')
		plt.show()

		# final evaluation metrics
		self.loss, self.accuracy = self._model.evaluate(self.validation)
		print(f'Model Accuracy: {self.accuracy}')


	'''
		================================
		
		================================
	'''	
	def predictions(self):
		# iterate over a few images
		count = 9
		
		for images, labels in self.validation.take(1):
		    
		    predicted_labels = np.argmax(self._model.predict(images), axis=-1)
		    
		    for i in range(count):
		        image = images[i].numpy()

		        true_label = self.classes[labels[i].numpy()]
		        predicted_label = self.classes[predicted_labels[i]]

		        self.display_prediction(image, true_label, predicted_label)


	'''
		================================
		
		================================
	'''	
	def display_prediction(self, image, true_label, predicted_label):
	    plt.figure()
	    plt.imshow(image.astype('uint8'))
	    plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
	    plt.axis('off')
	    plt.show()



	'''
		================================
		
		================================
	'''	
	def save(self):
		save_model(self._model, "../models/my_model.keras")
		

	'''
		================================
		
		================================
	'''	
	def load(self):
		self._model = load_model("../models/my_model.keras")
		print(self._model.summary())


# 
if __name__ == '__main__':
	# model instance
	_model = Captcha()

	# model setup/creation 
	_model.setup()
	# _model.create()

	# save the model
	# _model.save()

	# load the model
	_model.load()

	# 
	_model.predictions()