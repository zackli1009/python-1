# 🏆 2025年全国大学生光电设计竞赛 - 赛题A：激光对抗小车

> [cite_start]**硬件平台**: Jetson Orin Nano / NX   
> [cite_start]**核心组件**: PCA9685A 驱动舵机、12mm 长焦摄像头   
> [cite_start]**软件优化**: 原始代码 (224.py) + Claude 分块优化版本 

---

## [cite_start]🚩 第 1 部分 研究目标 

### **1. [cite_start]总体目标** 
#### [cite_start]设计并实现一套基于智能车的激光对抗系统，使智能车能够在竞赛场地中准确识别对手并进行有效“攻击”，同时具备躲避能力。 

### [cite_start]**1.1 功能目标** 
* #### **快速识别**：探测距离 **≥ 5 米**，覆盖范围达到 **180°**。 
* [cite_start]#### **高效躲避**：**2 秒内**完成转向/加减速动作，成功率 **≥ 80%**。 

### [cite_start]**1.2 性能目标** 
* #### **机动性**：直线速度 **≥ 1.5m/s**，转弯速度 **≥ 1m/s**。 
* [cite_start]#### **稳定性**：系统功耗控制在 **30W 以内**，持续运行 **20 分钟以上**。 

---

## [cite_start]🛠️ 第 2 部分 研究方案 

### **1. [cite_start]目标跟踪打击系统** 
#### [cite_start]**1.1 视觉算法 (YOLOv5)** 
#### [cite_start]使用 YOLOv5 配合 1000 张样本训练，实现在 6 米内精准识别。引入 **PAN 网络结构** 提高多角度识别率，并通过精调 **IOU 值** 优化精度。 

#### [cite_start]**1.2 跟踪方案** 
#### [cite_start]采用 **PID 算法** 控制 PWM 舵机，加入滤波算法减小抖动。结合 **交互式多模型 (IMM)** 与 **卡尔曼滤波**，在目标丢失时实现主动预测跟踪。 

#### [cite_start]**1.3 激光发射** 
#### [cite_start]选用半导体激光器配合高精度准直透镜。通过 PWM 电机驱动，实现发射方向的快速精确调整。 

### **2. [cite_start]路径规划与通信** 
* #### **避障控制**：基于 **STM32 + MPU6050** 实时控制方向，四路红外传感器避开雷区。 
* [cite_start]#### **串口通信**：Jetson 使用 Python **struct 库** 以数据包形式传输，配合 **UART 中断** 提高响应效率。 

---

## [cite_start]🔬 第 3 部分 技术路线及可行性分析 

* #### **硬件支撑**：市面成熟的光学传感器与底控芯片完全满足设计要求。 
* [cite_start]#### **软件基础**：YOLOv5 与路径规划算法已有成熟理论基础，具有高度可行性。 
* [cite_start]#### **团队实力**：成员具备电子电路、深度学习及自动控制的专业背景。 

---

## [cite_start]⚠️ 第 4 部分 拟解决的关键问题 

### **1. [cite_start]运动目标的精准识别** 
#### [cite_start]针对环境光干扰，通过优化传感器布局与改进信号处理算法来提升准确率。 

### **2. [cite_start]云台与激光瞄准** 
#### [cite_start]建立动力学模型，设计高效电机控制算法，确保在高速移动中也能精准打击。 

### **3. [cite_start]硬件连接优化** 
#### [cite_start]解决 Jetson 串口引脚冲突问题，采用 **TTL-USB 转接** 方式建立上位机与下位机的可靠通讯。 

---
---

---

<img width="2284" height="1157" alt="image" src="https://github.com/user-attachments/assets/121d1c7b-e952-443a-a5c1-ba97f174f48f" />
<img width="2366" height="1305" alt="image" src="https://github.com/user-attachments/assets/45ba5d87-509d-4858-b25b-737ca382dc49" />
<img width="2366" height="1305" alt="image" src="https://github.com/user-attachments/assets/dac564e3-e907-4c05-97e2-df104101bc8c" />



