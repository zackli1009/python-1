# ==================== config.py ====================
# 全局配置文件：所有常量、路径、PID参数集中在这里管理

# ----------- 路径配置 -----------
YOLOV5_REPO_PATH = '/home/orinnx/.local/share/Trash/files/yolov5'
MODEL_PT_PATH    = '/home/orinnx/zzx/yolov5-6-hutai/best2.pt'
BEEP_SCRIPT_PATH = '/home/orinnx/zzx/yolov5-6-hutai/led beep/beep.py'

# ----------- 摄像头 -----------
CAMERA_DEVICE_ID = 0

# ----------- 舵机初始位置 -----------
PAN_SERVO_CHANNEL  = 0
TILT_SERVO_CHANNEL = 1
PAN_INIT_ANGLE     = 110
TILT_INIT_ANGLE    = 45

# ----------- 模型置信度/IOU -----------
LARGE_OBJECT_CONF = 0.2
LARGE_OBJECT_IOU  = 0.2
SMALL_OBJECT_CONF = 0.45
SMALL_OBJECT_IOU  = 0.4

# ----------- 物体尺寸分类阈值 -----------
LARGE_OBJECT_THRESHOLD = 10000
SMALL_OBJECT_THRESHOLD = 8000

# ----------- 目标参数保持时间 -----------
PARAMETER_HOLD_TIME_AFTER_LOSS = 0.8   # 丢失目标后保持参数(s)
SWING_INITIAL_DELAY            = 1.1   # 丢失后延迟开始摆动(s)

# ----------- 云台摆动参数 -----------
SWING_STEP      = 1
SWING_MIN_ANGLE = 0
SWING_MAX_ANGLE = 175

# ----------- PID初始参数（无目标时重置用）-----------
PID_INIT = {
    'Kp_pan':           0.07,
    'Ki_pan':           0.001,
    'Kd_pan':           0.15,
    'Kp_tilt':          0.007,
    'Ki_tilt':          0.00007,
    'Kd_tilt':          0.0035,
    'max_output':       30,
    'max_angle_change': 1.5,
    'dead_zone':        15,
    'smooth_alpha':     0.5,
}

# ----------- 各场景PID参数组 -----------
PID_LARGE_SLOW = {
    'Kp_pan': 0.018, 'Ki_pan': 0.0012, 'Kd_pan': 0.1,
    'Kp_tilt': 0.0015, 'Ki_tilt': 0.00003, 'Kd_tilt': 0.0015,
    'max_output': 13, 'max_angle_change': 0.5, 'dead_zone': 15, 'smooth_alpha': 0.9,
}
PID_LARGE_FAST = {
    'Kp_pan': 0.03, 'Ki_pan': 0.0018, 'Kd_pan': 0.2,
    'Kp_tilt': 0.0015, 'Ki_tilt': 0.00003, 'Kd_tilt': 0.0015,
    'max_output': 18, 'max_angle_change': 0.8, 'dead_zone': 10, 'smooth_alpha': 0.8,
}
PID_SMALL_SLOW = {
    'Kp_pan': 0.015, 'Ki_pan': 0.001, 'Kd_pan': 0.12,
    'Kp_tilt': 0.002, 'Ki_tilt': 0.00004, 'Kd_tilt': 0.0018,
    'max_output': 12, 'max_angle_change': 0.5, 'dead_zone': 15, 'smooth_alpha': 0.9,
}
PID_SMALL_FAST = {
    'Kp_pan': 0.03, 'Ki_pan': 0.0015, 'Kd_pan': 0.2,
    'Kp_tilt': 0.002, 'Ki_tilt': 0.00004, 'Kd_tilt': 0.0025,
    'max_output': 20, 'max_angle_change': 0.9, 'dead_zone': 12, 'smooth_alpha': 0.7,
}

# ----------- 速度切换参数 -----------
LARGE_SPEED_HIGH  = 7
LARGE_SPEED_LOW   = 6
LARGE_SPEED_UPPER = 25
SMALL_SPEED_HIGH  = 4
SMALL_SPEED_LOW   = 3
SMALL_SPEED_UPPER = 13
MIN_HOLD_TIME     = 1.5   # 模式切换最小持续时间(s)
SPEED_HISTORY_LEN = 10

# ----------- 控制频率 -----------
CONTROL_INTERVAL = 0.0125  # 80Hz

# ----------- 云台输出阻尼系数 -----------
# 作用于 PID 输出到角度的缩放，降低可减少超调，提高可增加响应速度
SERVO_DAMPING = 0.8

# ----------- YOLOv5 推理尺寸 -----------
# YOLOv5 内部 stride=32，合法值：320 / 416 / 480 / 640
# 320: 最快，小目标精度略降；640: 最准，速度慢；建议先用 416 观察效果
INFER_SIZE = 416
