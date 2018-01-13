import argparse
import cv2
import time
from keras.models import load_model
import numpy as np
import os

def predict(img='images/test.jpg',v=False, image=None, filters=None):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	model = load_model('keypoint_detector.h5')
	save=False
	if v==False:
		if image is None:
			save=True
			image = cv2.imread(img)
		gray_image = cv2.cvtColor(cv2.cvtColor(np.copy(image), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
		faces = face_cascade.detectMultiScale(gray_image, 1.10, 4)
		for (x,y,w,h) in faces:
			cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 4)

		for i, (x,y,w,h) in enumerate(faces):
			cropped_image = gray_image[y:y+h, x:x+w]
			org_shape = cropped_image.shape
			cropped_image = cv2.resize(cropped_image, (96, 96)) / 255
			cropped_image = cropped_image.astype(np.float32)
			cropped_image = cropped_image.reshape(-1, 96, 96, 1)
			key_points = model.predict(cropped_image)
			xs = ((key_points[0][0::2] * 48 + 48)*org_shape[0]/96)+x
			ys = ((key_points[0][1::2] * 48 + 48)*org_shape[1]/96)+y
			if filters["key_points"]:
				for point in zip(xs,ys):
					cv2.circle(image, point, 4, (0,255,0), -1)
			if filters["glasses"]:
				sunglasses = cv2.imread("images/sunglasses.png", cv2.IMREAD_UNCHANGED)
				sunglass_width = int((xs[7]-xs[9])*1.1)
				sunglass_height = int((ys[10]-ys[8])/1.1)
				sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height))
				transparent_region = sunglass_resized[:,:,:3] != 0
				image[int(ys[9]):int(ys[9])+sunglass_height,int(xs[9]):int(xs[9])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
		if save:
			filename = os.path.basename(img).split('.')[0]
			cv2.imwrite(filename+"_detected.jpg", image)
			print("Image saved as {}_detected.jpg".format(filename))
		else:
			return image
	else:
		sunglasses = cv2.imread("images/sunglasses.png", cv2.IMREAD_UNCHANGED)
		# Create instance of video capturer
		cv2.namedWindow("face detection activated")
		vc = cv2.VideoCapture(0)
		# Try to get the first frame
		if vc.isOpened():
			rval, frame = vc.read()
		else:
			rval = False

		# keep video stream open
		try:
			while rval:
				gray_image = cv2.cvtColor(cv2.cvtColor(np.copy(frame), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
				faces = face_cascade.detectMultiScale(gray_image, 1.10, 4)

				for (x,y,w,h) in faces:
					cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 4)

				for i, (x,y,w,h) in enumerate(faces):
					cropped_image = gray_image[y:y+h, x:x+w]
					org_shape = cropped_image.shape
					cropped_image = cv2.resize(cropped_image, (96, 96)) / 255
					cropped_image = cropped_image.astype(np.float32)
					cropped_image = cropped_image.reshape(-1, 96, 96, 1)
					key_points = model.predict(cropped_image)
					xs = ((key_points[0][0::2] * 48 + 48)*org_shape[0]/96)+x
					ys = ((key_points[0][1::2] * 48 + 48)*org_shape[1]/96)+y
					for point in zip(xs,ys):
						cv2.circle(frame, point, 4, (0,255,0), -1)
					sunglass_width = int((xs[7]-xs[9])*1.1)
					sunglass_height = int((ys[10]-ys[8])/1.1)
					sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height))
					transparent_region = sunglass_resized[:,:,:3] != 0
					frame[int(ys[9]):int(ys[9])+sunglass_height,int(xs[9]):int(xs[9])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]

				# plot image from camera with detections marked
				cv2.imshow("face detection activated", frame)

				# exit functionality - press any key to exit laptop video
				key = cv2.waitKey(20)
				if key > 0: # exit by pressing any key
					# destroy windows
					cv2.destroyAllWindows()

					# hack from stack overflow for making sure window closes on osx --> https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
					for i in range (1,5):
						cv2.waitKey(1)
					return

				# read next frame
				time.sleep(0.05)             # control framerate for computation - default 20 frames per sec
				rval, frame = vc.read()
		except Exception as e:
			print(e)
			cv2.destroyAllWindows()

			# hack from stack overflow for making sure window closes on osx --> https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
			for i in range (1,5):
				cv2.waitKey(1)
			return

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', metavar='input-file', help='Pass the input image/video file location', action='store', type=argparse.FileType('r'), default='images/test.jpg')
	parser.add_argument('-v', type=bool, default=False, help='Set to True for webcam input default is False')
	results=parser.parse_args()
	predict(img=results.i.name, v=results.v, filters={"key_points": 1, "glasses": 1})

if __name__ == '__main__':
	main()
