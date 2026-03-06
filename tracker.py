# ==================== tracker.py ====================
# 卡尔曼滤波器封装：初始化、更新（检测到目标）、预测（目标丢失）

import cv2
import numpy as np


class TargetTracker:
    def __init__(self):
        self.kf = self._create_kalman_filter()
        self.is_initialized = False

    def _create_kalman_filter(self):
        """创建状态向量 [x, y, vx, vy] 的4状态2测量卡尔曼滤波器"""
        kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.array([
            [5e-2, 0,    0,    0   ],
            [0,    5e-2, 0,    0   ],
            [0,    0,    10.0, 0   ],
            [0,    0,    0,    10.0],
        ], dtype=np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        return kf

    def update(self, cx, cy):
        """
        检测到目标时调用：预测 + 校正。
        返回 (smooth_cx, smooth_cy, speed, predicted_cx, predicted_cy, pred_error_x, pred_error_y,
               corrected_cx, corrected_cy, corrected_vx, corrected_vy)
        首次调用时直接初始化滤波器，返回 speed=0。
        """
        if not self.is_initialized:
            self.kf.statePost = np.array(
                [[cx], [cy], [0], [0]], dtype=np.float32
            )
            self.is_initialized = True
            print("卡尔曼滤波器初始化成功")
            return cx, cy, 0.0, cx, cy, 0, 0, cx, cy, 0, 0

        # 预测
        prediction = self.kf.predict()
        predicted_cx = int(prediction[0, 0])
        predicted_cy = int(prediction[1, 0])
        predicted_vx = int(prediction[2, 0])
        predicted_vy = int(prediction[3, 0])

        pred_error_x = abs(predicted_cx - cx)
        pred_error_y = abs(predicted_cy - cy)

        # 校正
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        corrected = self.kf.correct(measurement)
        corrected_cx = int(corrected[0, 0])
        corrected_cy = int(corrected[1, 0])
        corrected_vx = int(corrected[2, 0])
        corrected_vy = int(corrected[3, 0])

        speed = float(np.sqrt(corrected_vx**2 + corrected_vy**2))

        # 根据预测误差决定使用滤波值还是原始检测值
        if pred_error_x < 50 and pred_error_y < 50:
            smooth_cx, smooth_cy = corrected_cx, corrected_cy
        else:
            smooth_cx, smooth_cy = cx, cy

        return (smooth_cx, smooth_cy, speed,
                predicted_cx, predicted_cy, pred_error_x, pred_error_y,
                corrected_cx, corrected_cy, corrected_vx, corrected_vy)

    def predict_only(self):
        """
        目标丢失时调用：仅预测，不校正。
        返回 (smooth_cx, smooth_cy, speed, predicted_cx, predicted_cy, predicted_vx, predicted_vy)
        """
        if not self.is_initialized:
            return 0, 0, 0.0, 0, 0, 0, 0

        prediction = self.kf.predict()
        predicted_cx = int(prediction[0, 0])
        predicted_cy = int(prediction[1, 0])
        predicted_vx = int(prediction[2, 0])
        predicted_vy = int(prediction[3, 0])
        speed = float(np.sqrt(predicted_vx**2 + predicted_vy**2))

        return predicted_cx, predicted_cy, speed, predicted_cx, predicted_cy, predicted_vx, predicted_vy
