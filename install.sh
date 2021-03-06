#!/bin/sh
echo "Installing the requirements ..."
pip install -r requirements.txt
cd darknet
echo "Compiling darknet ..."
make
mkdir weights
echo "Downloading the weights ..."
wget -P weights/ https://pjreddie.com/media/files/tiny-yolo.weights
cd ../src/
