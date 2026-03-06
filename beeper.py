# ==================== beeper.py ====================
# 蜂鸣器控制：用线程包装 subprocess 调用，避免阻塞主循环

import threading
import subprocess

import config as cfg


class Beeper:
    def __init__(self):
        self._thread: threading.Thread | None = None

    def alarm(self):
        """触发蜂鸣，如果上一次还在运行则跳过，避免重叠"""
        if self._thread is not None and self._thread.is_alive():
            print("蜂鸣器线程已在运行，跳过本次调用")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        print("启动蜂鸣器线程")
        try:
            result = subprocess.run(
                ['python3', cfg.BEEP_SCRIPT_PATH],
                capture_output=True,
                text=True,
                timeout=5,
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
            self._thread = None

    def wait(self, timeout: float = 2.0):
        """程序退出时等待蜂鸣器线程结束"""
        if self._thread is not None and self._thread.is_alive():
            print("等待蜂鸣器线程结束...")
            self._thread.join(timeout=timeout)
