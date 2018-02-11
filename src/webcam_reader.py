#!/usr/bin/env python
from v4l2capture import Video_device

class WebcamReader:
    def __init__(self):
        # Open the video device.
        self.device = Video_device("/dev/video0")

        # Suggest an image size to the device. The device may choose and
        # return another size if it doesn't support the suggested one.
        self.size_x, self.size_y = self.device.set_format(640, 480)

        print "device chose {0}x{1} res".format(self.size_x, self.size_y)

        # Create a buffer to store image data in. This must be done before
        # calling 'start' if v4l2capture is compiled with libv4l2. Otherwise
        # raises IOError.
        self.device.create_buffers(1)

        # Send the buffer to the device. Some devices require this to be done
        # before calling 'start'.
        self.device.queue_all_buffers()
