from utils import *
import numpy as np
import cv2
import os
from model import build_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import matplotlib
matplotlib.rcParams['interactive'] == True

def main():
	# Load training set
	X_train, y_train = load_data()
	print("X_train.shape == {}".format(X_train.shape))
	print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(y_train.shape, y_train.min(), y_train.max()))

	# Load testing set
	X_test, _ = load_data(test=True)
	print("X_test.shape == {}".format(X_test.shape))

	plot_data(X_train[0], y_train[0], trtest="train")
	model = build_model()

	print(model.summary())
	optimizer =  Adam(lr=0.001, decay=1e-6)
	#Compile model
	model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

	#Callbacks
	tensorboard = TensorBoard(log_dir='logs')
	
	model.fit(x=X_train, y=y_train, batch_size=128, epochs=1000, validation_split=0.2, callbacks=[tensorboard])
	
	model.save('keypoint_detector.h5')

	y_test = model.predict(X_test)

	plot_data(X_test[0], y_test[0], trtest="test")

if __name__ == '__main__':
	main()