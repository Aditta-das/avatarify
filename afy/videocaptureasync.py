# https://github.com/gilbertfrancois/video-capture-async

import threading
import cv2
import time


WARMUP_TIMEOUT = 10.0


class VideoCaptureAsync:
    def __init__(self, src=0, width=320, height=240): # width=640 height=480
        self.src = src
        self.faceCascade = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + self.faceCascade)
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 10) # set fps 10
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def isOpened(self):
        return self.cap.isOpened()

    def start(self):
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

        # (warmup) wait for the first successfully grabbed frame
        warmup_start_time = time.time()
        while not self.grabbed:
            warmup_elapsed_time = (time.time() - warmup_start_time)
            if warmup_elapsed_time > WARMUP_TIMEOUT:
                raise RuntimeError(f"Failed to succesfully grab frame from the camera (timeout={WARMUP_TIMEOUT}s). Try to restart.")

            time.sleep(0.5)

        return self

    def update(self):
        
        while self.started:
            grabbed, frame = self.cap.read()
            frame = cv2.flip(frame, 360) # flip 360 degree
            '''
            detect face using haarcascade
            '''
            faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # rectangle
                cropped_frame = frame[y:y+h, x:x+w]
                # cropped_frame = cv2.resize(cropped_frame,(320, 240))
                if not grabbed or frame is None or frame.size == 0:
                    continue
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = cropped_frame

    def read(self):
        while True:
            with self.read_lock:
                frame = self.frame.copy()
                grabbed = self.grabbed
            break
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
