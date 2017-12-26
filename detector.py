#!/usr/bin/env python

import sys, os
import math
sys.path.append(os.path.join(os.getcwd(),'darknet/python/'))
import glob
import darknet as dn
import numpy as np
import cv2

#Define the model, model's meta data and weights
model = "darknet/cfg/yolo.cfg"
metas = "darknet/cfg/coco.data"
weights = "darknet/weights/yolo.weights"
#model = "darknet/cfg/yolo9000.cfg"
#metas = "darknet/cfg/combine9k.data"
#weights = "darknet/weights/yolo9000.weights"

#load the net
net = dn.load_net(model, weights, 0)
meta = dn.load_meta(metas)

#get the images
imgs = glob.glob('/home/yassinel/Pictures/Webcam/*.jpg')

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

