import cv2

capture_device = 0
capture_fps = 30
capture_width = 640
capture_height = 480
width = 224
height = 224
format = 'bgr8'
running = False


def _gst_str():
    return 'nvarguscamerasrc sensor-id=%d ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
        capture_device, capture_width, capture_height, capture_fps, width, height)


cap = cv2.VideoCapture(_gst_str(), cv2.CAP_GSTREAMER)

re, image = cap.read()
