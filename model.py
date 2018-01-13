from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense

def build_model():

	model = Sequential()
	model.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(96,96,1)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.1))

	model.add(Convolution2D(filters=64, kernel_size=(2,2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(filters=128, kernel_size=(2,2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(units=500, activation='relu'))
	model.add(Dropout(0.6))
	model.add(Dense(units=500, activation='relu'))
	model.add(Dropout(0.8))
	model.add(Dense(units=30))

	return model