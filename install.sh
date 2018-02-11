#!/bin/sh
echo "Cloning the repository ..."
git clone https://github.com/yaselh/rt-yolo.git rt-yolo
echo "Installing the requirements ..."
pip install -r requirements.txt
cd rt-yolo/darknet
echo "Compiling darknet ..."
make
mkdir weights
echo "Downloading the weights ..."
wget -P weights https://pjreddie.com/media/files/tiny-yolo-voc.weights