#!/usr/bin/env python

import sys, os
import math
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
import numpy as np
import cv2
import glob
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Detect ojbects and draw the predicted Bounding Boxes')

	parser.add_argument('--imgs_path', type=str,   help='path to the images', required=True)
	parser.add_argument('--weights_path', type=str,   help='path to the weights file', default="weights/yolo.weights")
	parser.add_argument('--cfg_path', type=str,   help='path to a .cfg file', default="cfg/yolo.cfg")
	parser.add_argument('--data_path', type=str,   help='path to .data file', default="cfg/coco.data")

	args = parser.parse_args()

	#Define the model, model's meta data and weights
	model = args.cfg_path
	metas = args.data_path
	weights = args.weights_path
	#model = "cfg/yolo9000.cfg"
	#metas = "cfg/combine9k.data"
	#weights = "weights/yolo9000.weights"

	#load the net
	net = dn.load_net(model, weights, 0)
	meta = dn.load_meta(metas)

	#get the images
	imgs = glob.glob(args.imgs_path+'/*.jpg')

	#create the output directory
	if imgs:
		output_dir = os.path.dirname(imgs[0]) + "/predictions"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

	#perform the detection
	for img in imgs:
		#r = dn.detect(net, meta, img)
		output = output_dir + '/' + os.path.basename(img)[:-4]
		dn.test_detector(net, metas, img, .24, .5, output, False)
