#!/usr/bin/env python

import sys, os
import math
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
import numpy as np
import cv2
import glob

if __name__ == "__main__":
	if(len(sys.argv) < 2):
		print "Please provide the images directory"
		sys.exit()

	
	#Define the model, model's meta data and weights
	model = "cfg/yolo.cfg"
	metas = "cfg/coco.data"
	weights = "weights/yolo.weights"
	#model = "cfg/yolo9000.cfg"
	#metas = "cfg/combine9k.data"
	#weights = "weights/yolo9000.weights"

	#load the net
	net = dn.load_net(model, weights, 0)
	meta = dn.load_meta(metas)

	#get the images
	path = sys.argv[1]
	imgs = glob.glob(path+'/*.jpg')

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


