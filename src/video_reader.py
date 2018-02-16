#!/usr/bin/env python
import cv2
from time import time

class VideoReader(cv2.VideoCapture):
    def __init__(self):
        # Open the video device.
        cv2.VideoCapture.__init__(self, 0)

    def getGrabDuration(self):
        self.grab()
        before_grab = time()
        for i in range(5):
            self.grab()
        after_grab = time()
        return 1000*(after_grab - before_grab)/5

    def skipFrames(self, num_frames):
        before_skip = time()
        for i in range(num_frames):
            self.grab()
        after_skip = time()
        duration = after_skip - before_skip
        print("skipping {0} frames in {1:.2f} seconds".format(num_frames,duration))
