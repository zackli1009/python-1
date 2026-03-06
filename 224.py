import cv2
import torch
import time
from adafruit_servokit import ServoKit
import numpy as np
import subprocess
import threading


# 初始化云台
kit = ServoKit(channels=16)
pan_servo = 0
tilt_servo = 1
pan_angle = 110  # 初始水平角度（居中）
tilt_angle = 45  # 初始垂直角度（居中偏上）

# 设置舵机初始位置
try:
    kit.servo[pan_servo].angle = pan_angle
    kit.servo[tilt_servo].angle = tilt_angle
    time.sleep(1)
except OSError as e:
    print(f"舵机初始化失败: {e}")
    exit()

# 手动指定本地YOLOv5仓库路径
yolov5_path = '/home/orinnx/.local/share/Trash/files/yolov5'

# 加载YOLOv5模型（本地仓库方式）
try:
    model = torch.hub.load(yolov5_path, 'custom', path='/home/orinnx/zzx/yolov5-6-hutai/best2.pt', source='local')
    print("YOLOv5模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    kit._pca.deinit()
    exit()

# 模型参数设置
large_object_conf = 0.2
large_object_iou = 0.2
small_object_conf = 0.45
small_object_iou = 0.4
# 视频流初始化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    kit._pca.deinit()
    exit()

# 图像参数
img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
img_center = (img_width//2, img_height//2)

# 参数保持时间设置
PARAMETER_HOLD_TIME_AFTER_LOSS = 0.8  # 丢失目标后保持参数的时间(秒)
SWING_INITIAL_DELAY = 0.8          # 丢失目标后延迟摆动的时间(秒)

# 物体大小分类阈值
LARGE_OBJECT_THRESHOLD = 10000
SMALL_OBJECT_THRESHOLD = 8000

# ------------- PID控制参数 -------------
# 初始PID控制参数（无目标时）
Kp_pan_init = 0.07
Ki_pan_init = 0.001
Kd_pan_init = 0.15
Kp_tilt_init = 0.007
Ki_tilt_init = 0.00007
Kd_tilt_init = 0.0035
max_output_init = 30
max_angle_change_init = 1.5
dead_zone_init = 15
smooth_alpha_init = 0.5

# 大物体 - 慢速移动
large_slow = {
    'Kp_pan': 0.018,
    'Ki_pan': 0.0012,
    'Kd_pan': 0.1,
    'Kp_tilt': 0.0015,
    'Ki_tilt': 0.00003,
    'Kd_tilt': 0.0015,
    'max_output': 13,
    'max_angle_change': 0.5,
    'dead_zone': 15,
    'smooth_alpha': 0.9
}

# 大物体 - 快速移动
large_fast = {
    'Kp_pan': 0.03,
    'Ki_pan': 0.0018,
    'Kd_pan': 0.2,
    'Kp_tilt': 0.0015,
    'Ki_tilt': 0.00003,
    'Kd_tilt': 0.0015,
    'max_output': 18,
    'max_angle_change': 0.8,
    'dead_zone': 10,
    'smooth_alpha': 0.8
}

# 小物体 - 慢速移动
small_slow = {
    'Kp_pan': 0.015,  # 小物体慢速时稍微增加比例增益
    'Ki_pan': 0.001,
    'Kd_pan': 0.12,
    'Kp_tilt': 0.002,
    'Ki_tilt': 0.00004,
    'Kd_tilt': 0.0018,
    'max_output': 12,  # 小物体允许更大的输出
    'max_angle_change': 0.5,
    'dead_zone': 15,  # 小物体使用更小的死区
    'smooth_alpha': 0.9
}

# 小物体 - 快速移动
small_fast = {
    'Kp_pan': 0.03,  # 小物体快速时增加比例增益
    'Ki_pan': 0.0015,
    'Kd_pan': 0.2,
    'Kp_tilt': 0.002,
    'Ki_tilt': 0.00004,
    'Kd_tilt': 0.0025,
    'max_output': 20,  # 小物体快速时允许更大的输出
    'max_angle_change': 0.9,  # 小物体快速时允许更大的角度变化
    'dead_zone': 12,  # 小物体快速时使用更小的死区
    'smooth_alpha': 0.7  # 小物体快速时降低平滑度，增加响应速度
}

# 当前使用的PID参数
Kp_pan = Kp_pan_init
Ki_pan = Ki_pan_init
Kd_pan = Kd_pan_init
Kp_tilt = Kp_tilt_init
Ki_tilt = Ki_tilt_init
Kd_tilt = Kd_tilt_init
max_output = max_output_init
max_angle_change = max_angle_change_init
dead_zone = dead_zone_init
smooth_alpha = smooth_alpha_init

# ------------- 云台摆动参数 -------------
# 初始云台摆动参数（无目标时）
swing_params = {
    'step': 1,           # 每次摆动的角度步长
    'min_angle': 0,       # 最小水平角度（安全限制）
    'max_angle': 175,      # 最大水平角度（安全限制）
    'initial_delay': 1.1,  # 丢失目标后延迟开始摆动的时间
    'direction': 1,        # 摆动方向（1=右，-1=左）
    'enabled': False,      # 是否启用摆动
    'start_time': 0        # 摆动开始时间
}

# 状态变量
integral_pan = 0
last_error_pan = 0
integral_tilt = 0
last_error_tilt = 0
smooth_cx, smooth_cy = img_center
dx = dy = 0
last_cx, last_cy = img_center

# 计时器变量
detection_start_time = None
lost_start_time = None
near_center_start_time = None
last_detection_time = time.time()  # 上次检测到目标的时间

# 速度切换相关参数
# 大物体速度阈值
large_speed_threshold_high = 7  # 高速阈值（快速模式下限）
large_speed_threshold_low = 6   # 低速速阈值（慢速模式上限）
large_speed_upper_limit = 25   # 超速阈值（超过此值强制切换为慢速）
# 小物体速度阈值
small_speed_threshold_high = 4  # 高速阈值（快速模式下限）
small_speed_threshold_low = 3   # 低速速阈值（慢速模式上限）
small_speed_upper_limit = 13    # 超速阈值（超过此值强制切换为慢速）
min_hold_time = 1.5         # 模式切换最小持续时间（秒）
current_speed_mode = None
speed_mode_start_time = 0
current_object_size = None  # 记录当前物体大小分类

# 速度历史记录（用于平滑速度计算）
speed_history = []
speed_history_length = 10  # 记录最近10帧的速度

# 线程控制变量（蜂鸣器）
beep_thread = None

# 参数备份变量 - 初始化时包含所有可能的键
last_detected_params = {
    'Kp_pan': Kp_pan_init,
    'Ki_pan': Ki_pan_init,
    'Kd_pan': Kd_pan_init,
    'Kp_tilt': Kp_tilt_init,
    'Ki_tilt': Ki_tilt_init,
    'Kd_tilt': Kd_tilt_init,
    'max_output': max_output_init,
    'max_angle_change': max_angle_change_init,
    'dead_zone': dead_zone_init,
    'smooth_alpha': smooth_alpha_init,
    'object_size': 'unknown',  # 默认值
    'speed_mode': 'unknown'    # 默认值
}

# 初始化卡尔曼滤波器
def create_kalman_filter():
    """创建用于跟踪目标的卡尔曼滤波器"""
    # 状态向量: [x, y, vx, vy]
    kf = cv2.KalmanFilter(4, 2)
    
    # 状态转移矩阵 (离散化的匀速运动模型)
    dt = 1.0  # 时间间隔
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # 测量矩阵 (我们只能观测到位置)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    
    # 调整过程噪声协方差矩阵，增加对系统自身运动的适应性
    kf.processNoiseCov = np.array([
        [5e-2, 0, 0, 0],
        [0, 5e-2, 0, 0],
        [0, 0, 10.0, 0],
        [0, 0, 0, 10.0]
    ], dtype=np.float32)
    
    # 调整测量噪声协方差矩阵，增加对系统自身运动的适应性
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    
    # 后验误差协方差矩阵
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    
    return kf

# 初始化卡尔曼滤波器
kalman_filter = create_kalman_filter()
kalman_initialized = False

def pid_control(error, integral, last_error, Kp, Ki, Kd):
    """带滤波的PID控制器"""
    if abs(error) < dead_zone:
        return 0, integral, 0
    integral = np.clip(integral + error, -1000, 1000)  # 积分限幅，防止饱和
    derivative = error - last_error
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, derivative

def update_swing_state(is_detected, current_time):
    """更新云台摆动状态"""
    global swing_params
    
    if is_detected:
        # 检测到目标，停止摆动
        swing_params['enabled'] = False
    else:
        # 丢失目标
        if not swing_params['enabled']:
            # 首次丢失目标，记录开始时间
            swing_params['start_time'] = current_time
            swing_params['enabled'] = True
            print(f"目标丢失，开始计时摆动: {current_time}")

def apply_swing(current_time):
    """执行云台摆动逻辑"""
    global pan_angle, swing_params
    
    # 检查是否达到延迟时间
    if current_time - swing_params['start_time'] < swing_params['initial_delay']:
        return False  # 延迟期间，不执行摆动
    
    # 执行摆动
    pan_angle += swing_params['step'] * swing_params['direction']
    
    # 检查是否需要改变方向
    if pan_angle >= swing_params['max_angle']:
        pan_angle = swing_params['max_angle']  # 限制最大角度
        swing_params['direction'] = -1  # 反转方向
        print(f"云台摆动达到最大角度，反转方向: {current_time}")
    elif pan_angle <= swing_params['min_angle']:
        pan_angle = swing_params['min_angle']  # 限制最小角度
        swing_params['direction'] = 1  # 反转方向
        print(f"云台摆动达到最小角度，反转方向: {current_time}")
    
    return True  # 已执行摆动

# 应用基于物体大小和速度的PID参数
def apply_pid_params(object_size, speed_mode):
    global Kp_pan, Ki_pan, Kd_pan, Kp_tilt, Ki_tilt, Kd_tilt
    global max_output, max_angle_change, dead_zone, smooth_alpha
    
    if object_size == 'large':
        if speed_mode == 'fast':
            params = large_fast
        else:
            params = large_slow
    else:  # small
        if speed_mode == 'fast':
            params = small_fast
        else:
            params = small_slow
    
    # 应用参数
    Kp_pan = params['Kp_pan']
    Ki_pan = params['Ki_pan']
    Kd_pan = params['Kd_pan']
    Kp_tilt = params['Kp_tilt']
    Ki_tilt = params['Ki_tilt']
    Kd_tilt = params['Kd_tilt']
    max_output = params['max_output']
    max_angle_change = params['max_angle_change']
    dead_zone = params['dead_zone']
    smooth_alpha = params['smooth_alpha']

# 蜂鸣器控制函数 - 在单独线程中运行
def run_beep_script():
    """在独立线程中执行蜂鸣器脚本，避免阻塞主线程"""
    global beep_thread
    try:
        print("启动蜂鸣器线程")
        result = subprocess.run(
            ['python3', '/home/orinnx/zzx/yolov5-6-hutai/led beep/beep.py'],
            capture_output=True,
            text=True,
            timeout=5  # 设置超时时间，防止脚本卡死
        )
        if result.returncode != 0:
            print(f"蜂鸣器脚本执行失败: {result.stderr}")
        else:
            print(f"蜂鸣器脚本执行成功: {result.stdout}")
    except subprocess.TimeoutExpired:
        print("蜂鸣器脚本执行超时")
    except Exception as e:
        print(f"运行蜂鸣器脚本时出错: {e}")
    finally:
        beep_thread = None  # 重置线程变量

try:
    # 固定控制频率 (Hz)
    control_interval = 0.0125
    next_time = time.time()

    while True:
        speed = 0
        # 控制循环定时
        current_time = time.time()
        if current_time < next_time:
            time.sleep(max(0, next_time - current_time))
        next_time += control_interval

        # 读取帧数据
        ret, frame = cap.read()
        if not ret:
            print("视频流中断")
            break

        # 重置误差值
        dx = dy = 0

        # 目标检测
        results = model(frame)
        detections = results.pandas().xyxy[0]

        has_detected = False
        best_bbox = None
        detected_cx = 0
        detected_cy = 0
        object_size = 0

        if len(detections) > 0:
            has_detected = True
            last_detection_time = current_time
            
            # 选择置信度最高的目标
            best_idx = detections['confidence'].idxmax()
            x1, y1, x2, y2 = map(int, detections.iloc[best_idx][['xmin', 'ymin', 'xmax', 'ymax']])
            best_bbox = (x1, y1, x2, y2)
            detected_cx = (x1 + x2) // 2
            detected_cy = (y1 + y2) // 2

            # 计算检测物体的大小
            object_size = (x2 - x1) * (y2 - y1)
            
            # 确定物体大小分类
            if object_size > LARGE_OBJECT_THRESHOLD:
                current_object_size = 'large'
                # 根据物体大小调整 model.conf 和 model.iou
                model.conf = large_object_conf
                model.iou = large_object_iou
            elif object_size < SMALL_OBJECT_THRESHOLD:
                current_object_size = 'small'
                # 根据物体大小调整 model.conf 和 model.iou
                model.conf = small_object_conf
                model.iou = small_object_iou
            else:
                # 中等大小物体，根据之前状态决定
                if current_object_size is None:
                    current_object_size = 'small'  # 默认使用小物体参数
                    # 根据默认小物体参数调整 model.conf 和 model.iou
                    model.conf = small_object_conf
                    model.iou = small_object_iou
                # 否则保持之前的分类不变

            # 检测到目标时，备份当前参数
            last_detected_params = {
                'Kp_pan': Kp_pan,
                'Ki_pan': Ki_pan,
                'Kd_pan': Kd_pan,
                'Kp_tilt': Kp_tilt,
                'Ki_tilt': Ki_tilt,
                'Kd_tilt': Kd_tilt,
                'max_output': max_output,
                'max_angle_change': max_angle_change,
                'dead_zone': dead_zone,
                'smooth_alpha': smooth_alpha,
                'object_size': current_object_size,
                'speed_mode': current_speed_mode
            }

            # 初始化或更新卡尔曼滤波器
            if not kalman_initialized:
                # 第一次检测到目标，初始化卡尔曼滤波器
                kalman_filter.statePost = np.array([
                    [detected_cx],
                    [detected_cy],
                    [0],
                    [0]
                ], dtype=np.float32)
                kalman_initialized = True
                print("卡尔曼滤波器初始化成功")
            else:
                # 已有滤波器，进行预测和更新
                # 预测
                prediction = kalman_filter.predict()
                
                # 确保正确访问预测值
                predicted_cx = int(prediction[0, 0])
                predicted_cy = int(prediction[1, 0])
                predicted_vx = int(prediction[2, 0])
                predicted_vy = int(prediction[3, 0])
                
                # 计算预测误差
                pred_error_x = abs(predicted_cx - detected_cx)
                pred_error_y = abs(predicted_cy - detected_cy)
                
                # 更新
                measurement = np.array([[detected_cx], [detected_cy]], dtype=np.float32)
                corrected = kalman_filter.correct(measurement)
                
                # 确保正确访问修正值
                corrected_cx = int(corrected[0, 0])
                corrected_cy = int(corrected[1, 0])
                corrected_vx = int(corrected[2, 0])
                corrected_vy = int(corrected[3, 0])
                
                # 计算速度 (像素/帧)
                speed = np.sqrt(corrected_vx**2 + corrected_vy**2)
                
                # 使用卡尔曼滤波结果作为最终跟踪位置
                # 根据预测误差动态调整权重
                if pred_error_x < 50 and pred_error_y < 50:
                    # 预测误差较小，使用滤波后的结果
                    smooth_cx = corrected_cx
                    smooth_cy = corrected_cy
                else:
                    # 预测误差较大，更相信检测结果
                    smooth_cx = detected_cx
                    smooth_cy = detected_cy
                
                # 绘制预测位置和轨迹
                cv2.circle(frame, (predicted_cx, predicted_cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Prediction", (predicted_cx + 10, predicted_cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # 绘制速度向量（只显示明显的速度）
                if speed > 5:
                    cv2.arrowedLine(frame, (corrected_cx, corrected_cy), 
                                  (corrected_cx + int(corrected_vx*0.5), corrected_cy + int(corrected_vy*0.5)),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, f"V: {speed:.1f}px/f", (corrected_cx + 10, corrected_cy + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 计算中心偏移
            dx = int(smooth_cx - img_center[0])
            dy = int(smooth_cy - img_center[1])

            # 根据物体大小计算阈值和持续时间
            if object_size < 8000:
                near_center_threshold = 200
                duration_threshold = 1.3
            elif object_size > 12000:
                near_center_threshold = 600
                duration_threshold = 1.5
            else:
                # 平滑阈值过渡
                ratio = (object_size - 8000) / (12000 - 8000)
                near_center_threshold = 80 + ratio * (140 - 80)
                duration_threshold = 2 + ratio * (1.5 - 2)

            # 判断目标是否在中心点附近（触发蜂鸣器）
            if abs(dx) < near_center_threshold and abs(dy) < near_center_threshold:
                if near_center_start_time is None:
                    near_center_start_time = current_time
                elif current_time - near_center_start_time >= duration_threshold:
                    # 只在线程未运行时启动新线程
                    if beep_thread is None or not beep_thread.is_alive():
                        beep_thread = threading.Thread(target=run_beep_script)
                        beep_thread.daemon = True  # 守护线程，主程序退出时自动终止
                        beep_thread.start()
                        near_center_start_time = None
                    else:
                        print("蜂鸣器线程已在运行，跳过本次调用")
            else:
                near_center_start_time = None

            # 更新速度历史记录（用于平滑速度计算）
            speed_history.append(speed)
            if len(speed_history) > speed_history_length:
                speed_history.pop(0)

            # 计算平均速度（避免单帧抖动影响）
            average_speed = np.mean(speed_history) if speed_history else 0

            # 高速/低速参数切换逻辑
            if current_speed_mode is None:
                # 初始模式判断
                if current_object_size == 'large':
                    if (average_speed > large_speed_threshold_high) and (average_speed <= large_speed_upper_limit):
                        current_speed_mode = 'fast'
                    else:
                        current_speed_mode = 'slow'
                else:
                    if (average_speed > small_speed_threshold_high) and (average_speed <= small_speed_upper_limit):
                        current_speed_mode = 'fast'
                    else:
                        current_speed_mode = 'slow'
                speed_mode_start_time = current_time
                print(f"初始速度模式: {current_speed_mode} (平均速度: {average_speed:.1f})")
            else:
                elapsed_time = current_time - speed_mode_start_time
                can_switch = elapsed_time >= min_hold_time  # 满足最小持续时间才允许切换

                if can_switch:
                    if current_object_size == 'large':
                        # 快速模式 -> 慢速模式：速度<large_speed_threshold_low或速度>large_speed_upper_limit
                        if current_speed_mode == 'fast':
                            if (average_speed < large_speed_threshold_low) or (average_speed > large_speed_upper_limit):
                                current_speed_mode = 'slow'
                                speed_mode_start_time = current_time
                                print(f"切换到慢速模式 (平均速度: {average_speed:.1f}，原因：{'低于低速阈值' if average_speed < large_speed_threshold_low else '超过超速阈值'})")
                        
                        # 慢速模式 -> 快速模式：速度在(large_speed_threshold_high,large_speed_upper_limit]区间
                        elif current_speed_mode == 'slow':
                            if (average_speed > large_speed_threshold_high) and (average_speed <= large_speed_upper_limit):
                                current_speed_mode = 'fast'
                                speed_mode_start_time = current_time
                                print(f"切换到快速模式 (平均速度: {average_speed:.1f})")
                    else:
                        # 快速模式 -> 慢速模式：速度<small_speed_threshold_low或速度>small_speed_upper_limit
                        if current_speed_mode == 'fast':
                            if (average_speed < small_speed_threshold_low) or (average_speed > small_speed_upper_limit):
                                current_speed_mode = 'slow'
                                speed_mode_start_time = current_time
                                print(f"切换到慢速模式 (平均速度: {average_speed:.1f}，原因：{'低于低速阈值' if average_speed < small_speed_threshold_low else '超过超速阈值'})")
                        
                        # 慢速模式 -> 快速模式：速度在(small_speed_threshold_high,small_speed_upper_limit]区间
                        elif current_speed_mode == 'slow':
                            if (average_speed > small_speed_threshold_high) and (average_speed <= small_speed_upper_limit):
                                current_speed_mode = 'fast'
                                speed_mode_start_time = current_time
                                print(f"切换到快速模式 (平均速度: {average_speed:.1f})")

            # 应用基于物体大小和速度的PID参数
            apply_pid_params(current_object_size, current_speed_mode)

            # 显示当前速度模式和速度
            cv2.putText(frame, f"Obj Size: {current_object_size}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Speed Mode: {current_speed_mode}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Avg Speed: {average_speed:.1f}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 绘制检测框和目标中心
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {object_size:.1f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, (int(smooth_cx), int(smooth_cy)), 5, (0, 0, 255), -1)
            
            # 重置丢失计时
            lost_start_time = None
            
            # 更新摆动状态（停止摆动）
            update_swing_state(True, current_time)
        else:
            # 目标丢失，继续使用卡尔曼滤波器预测
            if kalman_initialized:
                prediction = kalman_filter.predict()
                
                # 确保正确访问预测值
                predicted_cx = int(prediction[0, 0])
                predicted_cy = int(prediction[1, 0])
                predicted_vx = int(prediction[2, 0])
                predicted_vy = int(prediction[3, 0])
                
                smooth_cx = predicted_cx
                smooth_cy = predicted_cy
                
                # 计算中心偏移
                dx = int(smooth_cx - img_center[0])
                dy = int(smooth_cy - img_center[1])
                
                # 计算速度
                speed = np.sqrt(predicted_vx**2 + predicted_vy**2)
                
                # 绘制预测位置和速度向量
                cv2.circle(frame, (predicted_cx, predicted_cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Prediction (Lost)", (predicted_cx + 10, predicted_cy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if speed > 5:
                    cv2.arrowedLine(frame, (predicted_cx, predicted_cy), 
                                  (predicted_cx + int(predicted_vx*0.5), predicted_cy + int(predicted_vy*0.5)),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, f"V: {speed:.1f}px/f", (predicted_cx + 10, predicted_cy + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # 重置速度模式状态和历史记录
            current_speed_mode = None
            detection_start_time = None
            near_center_start_time = None
            speed_history = []

            # 显示速度模式信息（无目标时）
            cv2.putText(frame, f"Speed Mode: None", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Speed: {speed:.1f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 目标丢失计时与参数恢复
            if lost_start_time is None:
                lost_start_time = current_time
                print(f"目标丢失，开始计时: {lost_start_time}")
            else:
                # 计算目标丢失时间
                lost_duration = current_time - lost_start_time
                
                # 在PARAMETER_HOLD_TIME_AFTER_LOSS秒内使用最后检测到的参数
                if lost_duration < PARAMETER_HOLD_TIME_AFTER_LOSS:
                    # 从备份中恢复参数
                    Kp_pan = last_detected_params['Kp_pan']
                    Ki_pan = last_detected_params['Ki_pan']
                    Kd_pan = last_detected_params['Kd_pan']
                    Kp_tilt = last_detected_params['Kp_tilt']
                    Ki_tilt = last_detected_params['Ki_tilt']
                    Kd_tilt = last_detected_params['Kd_tilt']
                    max_output = last_detected_params['max_output']
                    max_angle_change = last_detected_params['max_angle_change']
                    dead_zone = last_detected_params['dead_zone']
                    smooth_alpha = last_detected_params['smooth_alpha']
                    
                    # 安全获取object_size和speed_mode，避免KeyError
                    obj_size = last_detected_params.get('object_size', 'unknown')
                    speed_mode = last_detected_params.get('speed_mode', 'unknown')
                    
                    # 显示参数保持状态
                    cv2.putText(frame, f"Lost Target: {lost_duration:.1f}s", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Maintaining params...", (10, 210),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Size: {obj_size}", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Mode: {speed_mode}", (10, 270),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # 超过保持时间后恢复初始参数
                    if Kp_pan != Kp_pan_init:
                        Kp_pan = Kp_pan_init
                        Ki_pan = Ki_pan_init
                        Kd_pan = Kd_pan_init
                        Kp_tilt = Kp_tilt_init
                        Ki_tilt = Ki_tilt_init
                        Kd_tilt = Kd_tilt_init
                        max_output = max_output_init
                        max_angle_change = max_angle_change_init
                        dead_zone = dead_zone_init
                        smooth_alpha = smooth_alpha_init
                        print("恢复初始参数（目标丢失超过1秒）")
                    
                    # 显示搜索状态
                    cv2.putText(frame, f"Searching Target...", (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 更新摆动状态（开始摆动）
            update_swing_state(False, current_time)

        # ------------------- 云台跟踪逻辑 -------------------
        if has_detected or (kalman_initialized and current_time - lost_start_time < 5):
            # 水平PID计算
            pan_output, integral_pan, deriv_pan = pid_control(
                dx, integral_pan, last_error_pan, Kp_pan, Ki_pan, Kd_pan
            )
            # 垂直PID计算
            tilt_output, integral_tilt, deriv_tilt = pid_control(
                dy, integral_tilt, last_error_tilt, Kp_tilt, Ki_tilt, Kd_tilt
            )

            # 更新误差记录
            last_error_pan = dx
            last_error_tilt = dy

            # 输出限幅
            pan_output = np.clip(pan_output, -max_output, max_output)
            tilt_output = np.clip(tilt_output, -max_output, max_output)

            # 计算新角度（带阻尼）
            new_pan = pan_angle - pan_output * 0.8
            new_tilt = tilt_angle + tilt_output * 0.8

            # 渐进角度变化（限制最大角度变化量）
            pan_change = np.clip(new_pan - pan_angle, -max_angle_change, max_angle_change)
            tilt_change = np.clip(new_tilt - tilt_angle, -max_angle_change, max_angle_change)

            pan_angle += pan_change
            tilt_angle -= tilt_change

            # 安全角度限制
            pan_angle = np.clip(pan_angle, 0, 180)  # 水平角度安全范围
            tilt_angle = np.clip(tilt_angle, 35, 50)  # 垂直角度安全范围

            # 更新舵机位置
            try:
                kit.servo[pan_servo].angle = pan_angle
                kit.servo[tilt_servo].angle = tilt_angle
            except OSError as e:
                print(f"舵机控制异常: {e}")
        else:
            # 无目标时重置积分项
            integral_pan = integral_tilt = 0
            
            # 执行云台摆动
            if swing_params['enabled']:
                swing_executed = apply_swing(current_time)
                
                # 更新舵机位置
                if swing_executed:
                    try:
                        kit.servo[pan_servo].angle = pan_angle
                        # 显示摆动状态
                        cv2.putText(frame, f"Swinging: {pan_angle:.1f}deg", (10, 210),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    except OSError as e:
                        print(f"舵机控制异常: {e}")

        # 显示调试信息
        cv2.putText(frame, f"Pan: {pan_angle:.1f}deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Tilt: {tilt_angle:.1f}deg", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Error: ({dx}, {dy})", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Q: Quit", (img_width - 120, img_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 绘制中心标记
        cv2.drawMarker(frame, img_center, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

        # 显示画面
        cv2.imshow('AI Tracking System', frame)

        # 退出检测（按Q键）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户请求退出")
            break

except Exception as e:
    print(f"运行时异常: {e}")

finally:
    # 资源释放
    cap.release()
    cv2.destroyAllWindows()

    # 等待蜂鸣器线程结束
    if beep_thread is not None and beep_thread.is_alive():
        print("等待蜂鸣器线程结束...")
        beep_thread.join(timeout=2)  # 最多等待2秒

    # 舵机安全复位
    try:
        print("正在复位舵机...")
        kit.servo[pan_servo].angle = 110
        kit.servo[tilt_servo].angle = 45
        time.sleep(0.5)
        kit._pca.deinit()
    except Exception as e:
        print(f"资源释放异常: {e}")

    print("系统已安全关闭")
