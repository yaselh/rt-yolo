#!/usr/bin/env python
import os
import numpy as np
import cv2
import select
import time
from detector import Detector
from video_reader import VideoReader

class VideoDetector(Detector):
    def __init__(self):
        Detector.__init__(self)
        self.VideoReader = VideoReader()

    def detect(self, frame):
        frame_path = "frame.png"
        cv2.imwrite(frame_path, frame)
        result = Detector.detect(self, frame_path)
        os.remove(frame_path)
        return result

    def start(self):
        # Start the device. This lights the LED if it's a camera that has one.
        print "start capture"
        self.VideoReader.device.start()
        while(True):
            # Wait for the device to fill the buffer.
            select.select((self.VideoReader.device,), (), ())

            # Read frame
            image_data = self.VideoReader.device.read_and_queue()
            frame = np.frombuffer(image_data, dtype=np.uint8)
            frame = np.reshape(frame, (self.VideoReader.size_y,
                                       self.VideoReader.size_x,3))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect objects in frame
            start_time = time.time()
            detected_objects = self.detect(frame)
            end_time = time.time()
            delta_time = end_time - start_time
            print "Detection performed in " + "{0:.2f}".format(delta_time) + " seconds"
            print "Detected objects:"
            print detected_objects
            print "------------------------------------------------"
            # Draw bounding boxes in frame
            Detector.draw_bboxes(detected_objects, frame)

            # Show Frame
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        # Close device
        self.VideoReader.device.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = VideoDetector()
    detector.start()
