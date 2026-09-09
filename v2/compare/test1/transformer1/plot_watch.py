# -*- coding: utf-8 -*-
"""持续监视 training_log.txt，日志更新则重新绘制训练图表（复用 plot_training 逻辑）。

用法：
    python plot_watch.py [刷新间隔秒]

默认 20 秒一次。Ctrl+C 停止。
只读日志 + 重绘 PNG，不影响正在运行的程序。
"""
import contextlib
import io
import os
import sys
import time

import plot_training

DEFAULT_INTERVAL = 20.0


def _redraw():
    # 复用 plot_training.main()，抑制其冗长输出（重绘全部 PNG 到 plots/）
    with contextlib.redirect_stdout(io.StringIO()):
        plot_training.main()


def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INTERVAL
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, plot_training.LOG_FILE)

    print("持续绘图已启动：每 %.0f 秒检查一次日志。Ctrl+C 停止。" % interval)
    print("日志: %s" % log_path)

    last_size = -1
    while True:
        try:
            try:
                cur_size = os.path.getsize(log_path)
            except OSError:
                print("[%s] 日志不存在，等待..." % time.strftime('%H:%M:%S'))
                time.sleep(interval)
                continue

            if cur_size != last_size:
                last_size = cur_size
                print("\n[%s] 日志变化 (%d bytes)，重绘中..." % (time.strftime('%H:%M:%S'), cur_size))
                t0 = time.time()
                try:
                    _redraw()
                except Exception as e:
                    print("  绘图异常：%s（忽略）" % e)
                print("  完成，耗时 %.1fs" % (time.time() - t0))
            else:
                print("[%s] 日志未变化 (%d bytes)，跳过" % (time.strftime('%H:%M:%S'), cur_size))
        except KeyboardInterrupt:
            print("\n已停止。")
            return
        except Exception as e:
            print("  错误：%s" % e)
        time.sleep(interval)


if __name__ == "__main__":
    main()
