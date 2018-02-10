#!/usr/bin/env python

import sys, os
import math
darknet_path = "./darknet/"
sys.path.append(os.path.join(darknet_path, "python"))
import darknet as dn
import numpy as np
import glob
import argparse

class Detector:
	def __init__(self,model, metas, weights):
		#Define the model, model's meta data and weights
		self.model = model
		self.metas = metas
		self.weights = weights

		#load the net
		self.net = dn.load_net(model, weights, 0)
		self.meta = dn.load_meta(metas)

	def detect(self, img):
		return dn.detect(self.net, self.meta, img)

	def test(self, imgs_path):
		#get the images
		imgs = glob.glob(args.imgs_path+'/*.jpg')

		#create the output directory
		if imgs:
			output_dir = os.path.join(os.path.dirname(imgs[0]),"predictions")
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

		#perform the detection
		for img in imgs:
			r = self.detect(img)
			print r
			output = output_dir + '/' + os.path.basename(img)[:-4]
			#dn.test_detector(self.net, self.metas, img, .40, .5, output, False)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Detect ojbects and draw the predicted Bounding Boxes')

	parser.add_argument('--imgs_path', type=str,   help='path to the images', required=True)
	parser.add_argument('--weights_path', type=str,   help='path to the weights file', default="darknet/weights/yolo.weights")
	parser.add_argument('--cfg_path', type=str,   help='path to a .cfg file', default="darknet/cfg/yolo.cfg")
	parser.add_argument('--data_path', type=str,   help='path to .data file', default="darknet/cfg/coco.data")

	args = parser.parse_args()

	#Define the model, model's meta data and weights
	model = args.cfg_path #model = "cfg/yolo9000.cfg"
	metas = args.data_path #metas = "cfg/combine9k.data"
	weights = args.weights_path #weights = "weights/yolo9000.weights"

	detector = Detector(model, metas, weights)
	detector.test(args.imgs_path)
