# ==================== main.py ====================
# 主入口：编排 detector / tracker / gimbal / beeper，实现主控制循环

import cv2
import time
import numpy as np

import config as cfg
from camera   import ThreadedCamera
from detector import ObjectDetector
from tracker  import TargetTracker
from gimbal   import GimbalController
from beeper   import Beeper


def draw_debug(frame, pan_angle, tilt_angle, dx, dy,
               obj_size_str, speed_mode, avg_speed, img_width, img_height, img_center):
    """在帧上绘制调试信息"""
    cv2.putText(frame, f"Pan: {pan_angle:.1f}deg",       (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Tilt: {tilt_angle:.1f}deg",     (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Error: ({dx}, {dy})",            (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Obj Size: {obj_size_str}",       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Speed Mode: {speed_mode}",       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Avg Speed: {avg_speed:.1f}",     (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, "Q: Quit",
                (img_width - 120, img_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.drawMarker(frame, img_center, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)


def main():
    # ---- 初始化各模块 ----
    detector = ObjectDetector()
    tracker  = TargetTracker()
    gimbal   = GimbalController()
    beeper   = Beeper()

    try:
        cam = ThreadedCamera(cfg.CAMERA_DEVICE_ID)
    except RuntimeError as e:
        print(e)
        gimbal.shutdown()
        return

    img_width  = cam.img_width
    img_height = cam.img_height
    img_center = (img_width // 2, img_height // 2)

    # ---- 运行状态 ----
    smooth_cx, smooth_cy = img_center
    dx = dy = 0
    avg_speed = 0.0
    object_size = 0

    control_interval = cfg.CONTROL_INTERVAL
    next_time = time.time()

    try:
        while True:
            # 定频控制
            current_time = time.time()
            if current_time < next_time:
                time.sleep(max(0, next_time - current_time))
            next_time += control_interval

            ok, frame = cam.read()
            if not ok:
                print("视频流中断")
                break

            dx = dy = 0

            # ---- 目标检测 ----
            has_detected, best_bbox, detected_cx, detected_cy, object_size = detector.detect(frame)

            if has_detected:
                # 动态调整模型阈值，并拿到分类
                new_size_label = detector.adjust_thresholds(object_size, gimbal.current_object_size)
                gimbal.current_object_size = new_size_label

                # 卡尔曼更新
                (smooth_cx, smooth_cy, speed,
                 pred_cx, pred_cy, pred_err_x, pred_err_y,
                 corr_cx, corr_cy, corr_vx, corr_vy) = tracker.update(detected_cx, detected_cy)

                # 更新云台状态（速度模式 / PID 参数 / 丢失计时重置）
                avg_speed = gimbal.update_on_detection(object_size, speed, current_time)

                # 计算中心偏移
                dx = int(smooth_cx - img_center[0])
                dy = int(smooth_cy - img_center[1])

                # 蜂鸣器判断（仅在非首次帧，即卡尔曼已初始化至少一帧后）
                if gimbal.check_near_center(dx, dy, object_size, current_time):
                    beeper.alarm()

                # 绘制检测框和滤波结果
                x1, y1, x2, y2 = best_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Size: {object_size:.1f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (int(smooth_cx), int(smooth_cy)), 5, (0, 0, 255), -1)

                # 绘制卡尔曼预测点和速度向量（仅当滤波器已有历史时）
                if tracker.is_initialized:
                    cv2.circle(frame, (pred_cx, pred_cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "Prediction", (pred_cx + 10, pred_cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if speed > 5:
                        cv2.arrowedLine(frame,
                                        (corr_cx, corr_cy),
                                        (corr_cx + int(corr_vx * 0.5), corr_cy + int(corr_vy * 0.5)),
                                        (255, 0, 0), 2)
                        cv2.putText(frame, f"V: {speed:.1f}px/f",
                                    (corr_cx + 10, corr_cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            else:
                # ---- 目标丢失处理 ----
                lost_duration = gimbal.update_on_lost(current_time)

                if tracker.is_initialized:
                    (smooth_cx, smooth_cy, speed,
                     pred_cx, pred_cy, pred_vx, pred_vy) = tracker.predict_only()
                    dx = int(smooth_cx - img_center[0])
                    dy = int(smooth_cy - img_center[1])

                    cv2.circle(frame, (pred_cx, pred_cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "Prediction (Lost)", (pred_cx + 10, pred_cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if speed > 5:
                        cv2.arrowedLine(frame,
                                        (pred_cx, pred_cy),
                                        (pred_cx + int(pred_vx * 0.5), pred_cy + int(pred_vy * 0.5)),
                                        (255, 0, 0), 2)
                        cv2.putText(frame, f"V: {speed:.1f}px/f",
                                    (pred_cx + 10, pred_cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    speed = 0.0

                # 显示丢失状态
                if lost_duration < cfg.PARAMETER_HOLD_TIME_AFTER_LOSS:
                    obj_size_bak  = gimbal.last_detected_params.get('object_size', 'unknown')
                    speed_mode_bak = gimbal.last_detected_params.get('speed_mode', 'unknown')
                    cv2.putText(frame, f"Lost Target: {lost_duration:.1f}s",  (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, "Maintaining params...",                (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Size: {obj_size_bak}",                (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Mode: {speed_mode_bak}",              (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Searching Target...", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, "Speed Mode: None", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Speed: {speed:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ---- 云台控制逻辑 ----
            use_pid = has_detected or (
                tracker.is_initialized
                and gimbal.lost_start_time is not None
                and (current_time - gimbal.lost_start_time) < 5
            )

            if use_pid:
                pan_angle, tilt_angle = gimbal.track(dx, dy)
            else:
                gimbal.reset_integral()
                if gimbal.swing_params['enabled']:
                    swing_done = gimbal.do_swing(current_time)
                    if swing_done:
                        cv2.putText(frame,
                                    f"Swinging: {gimbal.pan_angle:.1f}deg",
                                    (10, 210),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                pan_angle  = gimbal.pan_angle
                tilt_angle = gimbal.tilt_angle

            # ---- 绘制调试信息 & 显示 ----
            draw_debug(frame, pan_angle, tilt_angle, dx, dy,
                       str(gimbal.current_object_size),
                       str(gimbal.current_speed_mode),
                       avg_speed,
                       img_width, img_height, img_center)

            cv2.imshow('AI Tracking System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户请求退出")
                break

    except Exception as e:
        print(f"运行时异常: {e}")

    finally:
        cam.stop()
        beeper.wait(timeout=2)
        gimbal.shutdown()
        cv2.destroyAllWindows()
        print("系统已安全关闭")


if __name__ == '__main__':
    main()
