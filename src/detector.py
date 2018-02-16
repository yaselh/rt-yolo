#!/usr/bin/env python

darknet_path = "../darknet/"
import sys, os
sys.path.append(os.path.join(darknet_path, "python"))
import darknet as dn
import math
import numpy as np
import glob
import argparse
import cv2
from time import time

class Detector:
	def __init__(self,model=darknet_path+"cfg/tiny-yolo.cfg",
					  metas=darknet_path+"cfg/coco.data",
					  weights=darknet_path+"weights/tiny-yolo.weights"):
		#Define the model, model's meta data and weights
		self.model = model
		self.metas = metas
		self.weights = weights

		#load the net
		self.net = dn.load_net(model, weights)
		self.meta = dn.load_meta(metas)

	@staticmethod
	def draw_bboxes(detected_objects, frame):
		bboxes = []
		for r in detected_objects:
		    cx,cy,w,h = r[2]
		    x, y = int(cx - w/2), int(cy - h/2)
		    h, w = int(h), int(w)

		    font = cv2.FONT_HERSHEY_SIMPLEX
		    cv2.putText(frame, r[0], (x+15,y+15), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
		    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

	def detect(self, img):
		start_time = time()
		r = dn.detect(self.net, self.meta, img)
		end_time = time()
		duration = end_time - start_time
		print("Detection performed in {0:.2f} seconds".format(duration))
		return r

	def test_detection(self, imgs_path):
		#get the images
		types = ('*.jpg', '*.jpeg', '*.png') # the tuple of file types
		imgs = []
		for files in types:
			imgs.extend(glob.glob(args.imgs_path+files))
		#create the output directory
		if imgs:
			output_dir = os.path.join(os.path.dirname(imgs[0]),"predictions")
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

		#perform the detection
		for img in imgs:
			#detect
			r = self.detect(img)
			print(r)
			#save
			output = output_dir + '/' + os.path.basename(img)
			img = cv2.imread(img)
			Detector.draw_bboxes(r, img)
			print(output)
			cv2.imwrite(output, img)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Detect ojbects and draw the predicted Bounding Boxes')

	parser.add_argument('--imgs_path', type=str,   help='path to the images', required=True)
	parser.add_argument('--weights_path', type=str,   help='path to the weights file', default=darknet_path+"weights/tiny-yolo.weights")
	parser.add_argument('--cfg_path', type=str,   help='path to a .cfg file', default=darknet_path+"cfg/tiny-yolo.cfg")
	parser.add_argument('--data_path', type=str,   help='path to .data file', default=darknet_path+"cfg/coco.data")

	args = parser.parse_args()

	#Define the model, model's meta data and weights
	model = args.cfg_path
	metas = args.data_path
	weights = args.weights_path

	detector = Detector(model, metas, weights)
	detector.test_detection(args.imgs_path)
