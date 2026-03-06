# ==================== detector.py ====================
# YOLOv5 模型封装：加载、推理、置信度阈值动态调整

import numpy as np
import torch

import config as cfg


class ObjectDetector:
    def __init__(self):
        try:
            self.model = torch.hub.load(
                cfg.YOLOV5_REPO_PATH,
                'custom',
                path=cfg.MODEL_PT_PATH,
                source='local',
            )
            print("YOLOv5模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

        # ---- 推理优化 ----
        # 1. 每次只保留置信度最高的 1 个检测框，减少 NMS 计算量
        self.model.max_det = 1

        # 2. FP16 半精度推理（Jetson Orin GPU 原生支持，可省 30-50% 显存和时间）
        #    若推理结果出现 nan，把下面这行注释掉退回 FP32
        if torch.cuda.is_available():
            self.model.half()
            self._use_half = True
        else:
            self._use_half = False

        # 3. 推理尺寸（YOLOv5 内部 stride=32，常用 320/416/640）
        #    值越小速度越快但小目标精度略降；原代码未显式设置，默认 640
        self._infer_size = cfg.INFER_SIZE

        # 初始默认阈值（先用小物体阈值）
        self.model.conf = cfg.SMALL_OBJECT_CONF
        self.model.iou  = cfg.SMALL_OBJECT_IOU

        # 4. Warm-up：用 dummy 帧跑一次，消除第一帧的 GPU 初始化延迟
        self._warmup()

    def _warmup(self):
        """用随机 dummy 帧预热 GPU，避免第一帧推理特别慢"""
        print("正在预热模型...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.model(dummy, size=self._infer_size)
            print("模型预热完成")
        except Exception as e:
            print(f"模型预热失败（不影响运行）: {e}")

    def detect(self, frame):
        """
        对一帧图像运行推理。
        返回:
            has_detection (bool)
            best_bbox     (x1, y1, x2, y2) | None
            detected_cx   int
            detected_cy   int
            object_size   int  (像素面积)

        优化说明：
        - 直接读 results.xyxy[0] tensor，避免每帧构造 pandas DataFrame
          （pandas 转换会触发 Python 层 for-loop 和字符串处理，约 1-3ms 额外开销）
        - 由于 max_det=1，xyxy[0] 最多只有 1 行，argmax 等价于直接取第 0 行
        """
        results = self.model(frame, size=self._infer_size)

        # xyxy[0]: shape (N, 6) = [x1, y1, x2, y2, conf, cls]，全部是 tensor
        preds = results.xyxy[0]

        if len(preds) == 0:
            return False, None, 0, 0, 0

        # max_det=1 时只有 1 行，直接取；多检测时取置信度最高的
        best = preds[preds[:, 4].argmax()]  # conf 在第 4 列

        x1, y1, x2, y2 = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        detected_cx = (x1 + x2) // 2
        detected_cy = (y1 + y2) // 2
        object_size = (x2 - x1) * (y2 - y1)

        return True, (x1, y1, x2, y2), detected_cx, detected_cy, object_size

    def adjust_thresholds(self, object_size: int, current_object_size: str | None) -> str:
        """
        根据物体面积像素值动态调整 model.conf / model.iou。
        返回更新后的 object_size 分类字符串 ('large' | 'small')。
        """
        if object_size > cfg.LARGE_OBJECT_THRESHOLD:
            self.model.conf = cfg.LARGE_OBJECT_CONF
            self.model.iou  = cfg.LARGE_OBJECT_IOU
            return 'large'
        elif object_size < cfg.SMALL_OBJECT_THRESHOLD:
            self.model.conf = cfg.SMALL_OBJECT_CONF
            self.model.iou  = cfg.SMALL_OBJECT_IOU
            return 'small'
        else:
            if current_object_size is None:
                self.model.conf = cfg.SMALL_OBJECT_CONF
                self.model.iou  = cfg.SMALL_OBJECT_IOU
                return 'small'
            return current_object_size
