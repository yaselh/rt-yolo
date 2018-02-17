#!/usr/bin/env python

import os
import numpy as np
import cv2
import select
import threading
from time import time
from detector import Detector
from video_reader import VideoReader

class VideoDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.videoReader = VideoReader()
        self.detectedObjects = []

    def detect(self, frame):
        frame_path = "frame.png"
        cv2.imwrite(frame_path, frame)
        self.detectedObjects = Detector.detect(self, frame_path)
        #os.remove(frame_path)
        return self.detectedObjects

    def start(self):
        print("start capture ...")
        while(True):
            # Read frame
            ret, frame = self.videoReader.read()

            # Detect objects in frame
            detect = threading.Thread(target=VideoDetector.detect,args=(self,frame))
            detect.start()

            # read and show frames without detecting while detection is being performed
            while detect.isAlive():
                ret, frame = self.videoReader.read()

                # Draw bounding boxes in frame
                Detector.draw_bboxes(self.detectedObjects, frame)

                # Show frame
                cv2.imshow('Detection',frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Close device
                    self.videoReader.release()
                    cv2.destroyAllWindows()
                    return


if __name__ == '__main__':
    detector = VideoDetector()
    detector.start()
