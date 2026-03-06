# ==================== camera.py ====================
# 独立线程读取摄像头帧，主线程直接拿最新帧，避免 cap.read() 阻塞推理。

import cv2
import threading


class ThreadedCamera:
    def __init__(self, device_id: int = 0):
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 device_id={device_id}")

        self.img_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("摄像头初始读帧失败")
        self._frame = frame
        self._ok    = True
        self._lock  = threading.Lock()
        self._stop  = False

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if not ret:
                self._ok = False
                break
            with self._lock:
                self._frame = frame

    def read(self):
        """返回 (ok, frame)。frame 是最新帧，不阻塞。"""
        with self._lock:
            return self._ok, self._frame.copy()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1)
        self.cap.release()
