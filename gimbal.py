# ==================== gimbal.py ====================
# 云台控制：舵机初始化、PID计算、角度输出、摆动逻辑、参数切换

import time
import numpy as np
from collections import deque
from adafruit_servokit import ServoKit

import config as cfg


def _pick_pid(object_size: str, speed_mode: str) -> dict:
    """根据物体大小和速度模式选择对应的PID参数字典"""
    table = {
        ('large', 'fast'): cfg.PID_LARGE_FAST,
        ('large', 'slow'): cfg.PID_LARGE_SLOW,
        ('small', 'fast'): cfg.PID_SMALL_FAST,
        ('small', 'slow'): cfg.PID_SMALL_SLOW,
    }
    return table.get((object_size, speed_mode), cfg.PID_INIT)


class GimbalController:
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.pan_servo  = cfg.PAN_SERVO_CHANNEL
        self.tilt_servo = cfg.TILT_SERVO_CHANNEL

        self.pan_angle  = cfg.PAN_INIT_ANGLE
        self.tilt_angle = cfg.TILT_INIT_ANGLE

        # 初始化舵机位置
        try:
            self.kit.servo[self.pan_servo].angle  = self.pan_angle
            self.kit.servo[self.tilt_servo].angle = self.tilt_angle
            time.sleep(1)
        except OSError as e:
            print(f"舵机初始化失败: {e}")
            raise

        # --- 当前生效的PID参数（从 PID_INIT 展开）---
        self._load_pid(cfg.PID_INIT)

        # PID状态
        self.integral_pan    = 0.0
        self.integral_tilt   = 0.0
        self.last_error_pan  = 0.0
        self.last_error_tilt = 0.0
        self.derivative_pan  = 0.0  # 滤波后的微分项（persistent）
        self.derivative_tilt = 0.0

        # 速度/尺寸模式
        self.current_object_size = None
        self.current_speed_mode  = None
        self.speed_mode_start_time = 0.0
        self.speed_history: deque[float] = deque(maxlen=cfg.SPEED_HISTORY_LEN)

        # 舵机角度缓存（去重：变化量 < 阈值时不写 I2C）
        self._last_pan_sent  = self.pan_angle
        self._last_tilt_sent = self.tilt_angle
        self._servo_threshold = 0.2  # 度

        # 参数备份（目标丢失短暂保持用）
        self.last_detected_params = dict(cfg.PID_INIT)
        self.last_detected_params.update({'object_size': 'unknown', 'speed_mode': 'unknown'})

        # 时间戳
        self.lost_start_time      = None
        self.near_center_start_time = None

        # 云台摆动
        self.swing_params = {
            'step': cfg.SWING_STEP,
            'min_angle': cfg.SWING_MIN_ANGLE,
            'max_angle': cfg.SWING_MAX_ANGLE,
            'initial_delay': cfg.SWING_INITIAL_DELAY,
            'direction': 1,
            'enabled': False,
            'start_time': 0.0,
        }

    # ------------------------------------------------------------------ #
    #  内部工具                                                            #
    # ------------------------------------------------------------------ #
    def _load_pid(self, params: dict):
        """把一个 PID 参数字典写入实例属性"""
        self.Kp_pan          = params['Kp_pan']
        self.Ki_pan          = params['Ki_pan']
        self.Kd_pan          = params['Kd_pan']
        self.Kp_tilt         = params['Kp_tilt']
        self.Ki_tilt         = params['Ki_tilt']
        self.Kd_tilt         = params['Kd_tilt']
        self.max_output      = params['max_output']
        self.max_angle_change = params['max_angle_change']
        self.dead_zone       = params['dead_zone']
        self.smooth_alpha    = params['smooth_alpha']

    def _pid_control(self, error, integral, last_error, derivative_prev,
                     Kp, Ki, Kd, smooth_alpha):
        """
        带以下改进的 PID 控制器：
        1. 死区：误差 < dead_zone 直接返回 0
        2. D项低通滤波：用 smooth_alpha 对 derivative 进行 EMA 平滑
           (smooth_alpha→1 = 响应快/少滤波, →0 = 响应慢/多滤波)
        3. Anti-windup：输出饱和时不再累积积分（条件积分）
        """
        if abs(error) < self.dead_zone:
            return 0.0, integral, derivative_prev

        # 微分项：EMA低通滤波，消除帧间抖动对D项的放大
        raw_derivative = error - last_error
        derivative = smooth_alpha * raw_derivative + (1.0 - smooth_alpha) * derivative_prev

        # 计算未限幅的输出（用于 anti-windup 判断）
        output_unclipped = Kp * error + Ki * integral + Kd * derivative

        # Anti-windup：仅当输出未饱和，或积分方向与误差反向时才更新积分
        # （避免已饱和时积分继续朝饱和方向累积）
        at_limit = abs(output_unclipped) >= self.max_output
        same_sign = (error * integral) > 0  # 误差和积分同号 → 积分在往错误方向走
        if not (at_limit and same_sign):
            integral = float(np.clip(integral + error, -1000, 1000))

        output = Kp * error + Ki * integral + Kd * derivative
        return output, integral, derivative

    # ------------------------------------------------------------------ #
    #  速度模式切换                                                         #
    # ------------------------------------------------------------------ #
    def _update_speed_mode(self, average_speed: float, current_time: float):
        size = self.current_object_size
        if size == 'large':
            hi, lo, upper = cfg.LARGE_SPEED_HIGH, cfg.LARGE_SPEED_LOW, cfg.LARGE_SPEED_UPPER
        else:
            hi, lo, upper = cfg.SMALL_SPEED_HIGH, cfg.SMALL_SPEED_LOW, cfg.SMALL_SPEED_UPPER

        if self.current_speed_mode is None:
            # 初次判断
            if hi < average_speed <= upper:
                self.current_speed_mode = 'fast'
            else:
                self.current_speed_mode = 'slow'
            self.speed_mode_start_time = current_time
            print(f"初始速度模式: {self.current_speed_mode} (平均速度: {average_speed:.1f})")
            return

        if current_time - self.speed_mode_start_time < cfg.MIN_HOLD_TIME:
            return  # 最小持续时间内不切换

        if self.current_speed_mode == 'fast':
            if average_speed < lo or average_speed > upper:
                reason = '低于低速阈值' if average_speed < lo else '超过超速阈值'
                self.current_speed_mode = 'slow'
                self.speed_mode_start_time = current_time
                print(f"切换到慢速模式 (平均速度: {average_speed:.1f}，原因：{reason})")
        else:  # slow
            if hi < average_speed <= upper:
                self.current_speed_mode = 'fast'
                self.speed_mode_start_time = current_time
                print(f"切换到快速模式 (平均速度: {average_speed:.1f})")

    # ------------------------------------------------------------------ #
    #  公开接口                                                             #
    # ------------------------------------------------------------------ #
    def update_on_detection(self, object_size_px: int, speed: float, current_time: float):
        """每帧检测到目标时调用，更新尺寸分类、速度模式、PID参数"""
        # 尺寸分类
        if object_size_px > cfg.LARGE_OBJECT_THRESHOLD:
            self.current_object_size = 'large'
        elif object_size_px < cfg.SMALL_OBJECT_THRESHOLD:
            self.current_object_size = 'small'
        else:
            if self.current_object_size is None:
                self.current_object_size = 'small'

        # 速度历史（deque 自动维护 maxlen，O(1) append）
        self.speed_history.append(speed)
        average_speed = float(np.mean(self.speed_history)) if self.speed_history else 0.0

        # 速度模式切换
        self._update_speed_mode(average_speed, current_time)

        # 应用PID参数
        prev_speed_mode = getattr(self, '_prev_speed_mode', None)
        self._load_pid(_pick_pid(self.current_object_size, self.current_speed_mode))

        # 速度模式切换时清空 last_error，避免 D 项产生冲击
        if self.current_speed_mode != prev_speed_mode and prev_speed_mode is not None:
            self.last_error_pan  = 0.0
            self.last_error_tilt = 0.0
            self.derivative_pan  = 0.0
            self.derivative_tilt = 0.0
        self._prev_speed_mode = self.current_speed_mode

        # 备份当前参数（目标丢失后短暂保持用）
        self.last_detected_params = {
            'Kp_pan': self.Kp_pan, 'Ki_pan': self.Ki_pan, 'Kd_pan': self.Kd_pan,
            'Kp_tilt': self.Kp_tilt, 'Ki_tilt': self.Ki_tilt, 'Kd_tilt': self.Kd_tilt,
            'max_output': self.max_output, 'max_angle_change': self.max_angle_change,
            'dead_zone': self.dead_zone, 'smooth_alpha': self.smooth_alpha,
            'object_size': self.current_object_size,
            'speed_mode': self.current_speed_mode,
        }

        # 停止摆动，重置丢失计时
        self.swing_params['enabled'] = False
        self.lost_start_time = None

        return average_speed

    def update_on_lost(self, current_time: float):
        """每帧目标丢失时调用，处理参数保持/恢复，触发摆动"""
        # 重置速度状态
        self.current_speed_mode = None
        self.near_center_start_time = None
        self.speed_history.clear()

        # 丢失计时
        if self.lost_start_time is None:
            self.lost_start_time = current_time
            print(f"目标丢失，开始计时: {self.lost_start_time}")
            lost_duration = 0.0
        else:
            lost_duration = current_time - self.lost_start_time

        if lost_duration < cfg.PARAMETER_HOLD_TIME_AFTER_LOSS:
            # 短暂保持上次检测到时的参数
            self._load_pid(self.last_detected_params)
        else:
            # 超时后重置为初始参数
            if self.Kp_pan != cfg.PID_INIT['Kp_pan']:
                self._load_pid(cfg.PID_INIT)
                print("恢复初始参数（目标丢失超出保持时间）")

        # 启用摆动计时
        if not self.swing_params['enabled']:
            self.swing_params['start_time'] = current_time
            self.swing_params['enabled'] = True
            print(f"目标丢失，开始计时摆动: {current_time}")

        return lost_duration

    def check_near_center(self, dx: int, dy: int, object_size_px: int, current_time: float) -> bool:
        """判断目标是否持续处于中心区域，返回是否应触发蜂鸣"""
        if object_size_px < 8000:
            threshold = 200
            duration  = 1.3
        elif object_size_px > 12000:
            threshold = 600
            duration  = 1.5
        else:
            ratio = (object_size_px - 8000) / (12000 - 8000)
            threshold = int(80 + ratio * (140 - 80))
            duration  = 2 + ratio * (1.5 - 2)

        if abs(dx) < threshold and abs(dy) < threshold:
            if self.near_center_start_time is None:
                self.near_center_start_time = current_time
            elif current_time - self.near_center_start_time >= duration:
                self.near_center_start_time = None
                return True
        else:
            self.near_center_start_time = None
        return False

    def track(self, dx: int, dy: int):
        """根据误差进行PID计算并输出到舵机。返回 (pan_angle, tilt_angle)"""
        pan_output, self.integral_pan, self.derivative_pan = self._pid_control(
            dx, self.integral_pan, self.last_error_pan, self.derivative_pan,
            self.Kp_pan, self.Ki_pan, self.Kd_pan, self.smooth_alpha
        )
        tilt_output, self.integral_tilt, self.derivative_tilt = self._pid_control(
            dy, self.integral_tilt, self.last_error_tilt, self.derivative_tilt,
            self.Kp_tilt, self.Ki_tilt, self.Kd_tilt, self.smooth_alpha
        )

        self.last_error_pan  = dx
        self.last_error_tilt = dy

        pan_output  = float(np.clip(pan_output,  -self.max_output, self.max_output))
        tilt_output = float(np.clip(tilt_output, -self.max_output, self.max_output))

        damping = cfg.SERVO_DAMPING
        new_pan  = self.pan_angle  - pan_output  * damping
        new_tilt = self.tilt_angle + tilt_output * damping

        pan_change  = float(np.clip(new_pan  - self.pan_angle,  -self.max_angle_change, self.max_angle_change))
        tilt_change = float(np.clip(new_tilt - self.tilt_angle, -self.max_angle_change, self.max_angle_change))

        self.pan_angle  += pan_change
        self.tilt_angle -= tilt_change

        self.pan_angle  = float(np.clip(self.pan_angle,  0,  180))
        self.tilt_angle = float(np.clip(self.tilt_angle, 35,  50))

        try:
            if abs(self.pan_angle - self._last_pan_sent) >= self._servo_threshold:
                self.kit.servo[self.pan_servo].angle = self.pan_angle
                self._last_pan_sent = self.pan_angle
            if abs(self.tilt_angle - self._last_tilt_sent) >= self._servo_threshold:
                self.kit.servo[self.tilt_servo].angle = self.tilt_angle
                self._last_tilt_sent = self.tilt_angle
        except OSError as e:
            print(f"舵机控制异常: {e}")

        return self.pan_angle, self.tilt_angle

    def do_swing(self, current_time: float) -> bool:
        """执行摆动，返回是否真正执行了摆动"""
        if not self.swing_params['enabled']:
            return False
        if current_time - self.swing_params['start_time'] < self.swing_params['initial_delay']:
            return False

        self.pan_angle += self.swing_params['step'] * self.swing_params['direction']

        if self.pan_angle >= self.swing_params['max_angle']:
            self.pan_angle = self.swing_params['max_angle']
            self.swing_params['direction'] = -1
            print(f"云台摆动达到最大角度，反转方向: {current_time}")
        elif self.pan_angle <= self.swing_params['min_angle']:
            self.pan_angle = self.swing_params['min_angle']
            self.swing_params['direction'] = 1
            print(f"云台摆动达到最小角度，反转方向: {current_time}")

        try:
            self.kit.servo[self.pan_servo].angle = self.pan_angle
        except OSError as e:
            print(f"舵机控制异常: {e}")
        return True

    def reset_integral(self):
        """无目标时重置PID积分项"""
        self.integral_pan = self.integral_tilt = 0.0

    def shutdown(self):
        """安全复位舵机并释放硬件资源"""
        try:
            print("正在复位舵机...")
            self.kit.servo[self.pan_servo].angle  = cfg.PAN_INIT_ANGLE
            self.kit.servo[self.tilt_servo].angle = cfg.TILT_INIT_ANGLE
            time.sleep(0.5)
            self.kit._pca.deinit()
        except Exception as e:
            print(f"资源释放异常: {e}")
