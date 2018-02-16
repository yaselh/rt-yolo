#!/usr/bin/env python

import os
import numpy as np
import cv2
import select
from multiprocessing.pool import ThreadPool
from time import time
from detector import Detector
from video_reader import VideoReader

class VideoDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.videoReader = VideoReader()
        self.detectorPool = ThreadPool(processes=1)
        self.readerPool = ThreadPool(processes=4)

    def detect(self, frame):
        frame_path = "frame.png"
        cv2.imwrite(frame_path, frame)
        result = Detector.detect(self, frame_path)
        os.remove(frame_path)
        return result

    def start(self):
        t1, t2 = None,None
        grab_duration = int(self.videoReader.getGrabDuration())
        num_skips = int(1800/(1.5*grab_duration))

        print("start capture ...")
        while(True):
            # Read frame
            ret, frame = self.videoReader.read()

            s = time()

            # Detect objects in frame
            detect = self.detectorPool.apply_async(VideoDetector.detect, (self,frame))

            # Skip some frames while detecting objects
            self.readerPool.apply_async(VideoReader.skipFrames, (self.videoReader, num_skips))

            # Draw bounding boxes in frame
            Detector.draw_bboxes(detect.get(), frame)

            e = time()
            print("Total duration {0:.2f} seconds".format(e-s))
            # Show frame
            cv2.imshow('Detection',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Close device
        self.videoReader.release()
        cv2.destroyAllWindows()
    '''
    def start(self):
        # Start the device. This lights the LED if it's a camera that has one.
        print("start capture")
        self.videoReader.start()
        while(True):
            # Wait for the device to fill the buffer.
            select.select((self.videoReader,), (), ())

            # Read frame
            image_data = self.videoReader.read_and_queue()
            frame = np.frombuffer(image_data, dtype=np.uint8)
            frame = np.reshape(frame, (self.videoReader.size_y,
                                       self.videoReader.size_x,3))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect objects in frame
            start_time = time()
            detected_objects = self.detect(frame)
            end_time = time()
            delta_time = end_time - start_time
            print("Detection performed in " + "{0:.2f}".format(delta_time) + " seconds")
            print("Detected objects:")
            print(detected_objects)
            print("------------------------------------------------")
            # Draw bounding boxes in frame
            Detector.draw_bboxes(detected_objects, frame)

            # Show Frame
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        # Close device
        self.videoReader.close()
        cv2.destroyAllWindows()
    '''

if __name__ == '__main__':
    detector = VideoDetector()
    detector.start()
