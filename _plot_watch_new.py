# -*- coding: utf-8 -*-
"""持续绘图：日志更新则重绘（调用 plot_training 的绘图逻辑）。"""
import os, sys, time
sys.path.insert(0, r"D:\dataset\ultra6")
import plot_training

DEFAULT_INTERVAL = 20.0

def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INTERVAL
    script_dir = r"D:\dataset\ultra6"
    log_path = os.path.join(script_dir, plot_training.LOG_FILE)
    print("持续绘图: 每 %s 秒检查日志, Ctrl+C 停止" % interval)
    print("日志: %s" % log_path)
    last_size = -1
    while True:
        try:
            try:
                cur_size = os.path.getsize(log_path)
            except OSError:
                print("[%s] 日志不存在, 等待..." % time.strftime("%H:%M:%S")); time.sleep(interval); continue
            if cur_size != last_size:
                last_size = cur_size
                print("\n[%s] 日志变化 (%d bytes), 重绘中..." % (time.strftime("%H:%M:%S"), cur_size))
                t0 = time.time()
                try:
                    plot_training.main()
                except Exception as e:
                    print("  绘图异常: %s (忽略)" % e)
                print("  完成, 耗时 %.1fs" % (time.time()-t0))
            else:
                print("[%s] 日志未变化 (%d bytes), 跳过" % (time.strftime("%H:%M:%S"), cur_size))
        except KeyboardInterrupt:
            print("\n已停止."); return
        except Exception as e:
            print("  错误: %s" % e)
        time.sleep(interval)

if __name__ == "__main__":
    main()