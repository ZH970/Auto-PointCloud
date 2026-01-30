from pynput.keyboard import Listener, Key
from datetime import datetime, timedelta

# 全局变量：记录上一次按下的按键，初始化为 None
last_key = None
last_enter_time = None
press_status = 0  # 用于记录是否检测到连续两次回车

def on_press(key):
    """
    键盘按下事件回调函数
    """
    global last_key, last_enter_time, press_status
    current_key = None
    
    # 1. 判断当前按键是否是回车键（Enter）
    try:
        # 普通字符按键（这里用不到，仅兼容）
        current_key = key.char
    except AttributeError:
        # 特殊按键（回车、空格等）
        if key == Key.enter:
            current_key = "enter"  # 用字符串标记回车键
    
    # 2. 核心逻辑：判断是否是连续两次按下回车

    if current_key == "enter":
        current_time = datetime.now()
        if last_key == "enter" and last_enter_time:
            # 计算两次回车的时间间隔（秒）
            interval = (current_time - last_enter_time).total_seconds()
            if interval <= 1:  # 间隔不超过 1 秒视为连续
                print("\n检测到连续两次按下回车，返回 1")
                press_status = 1
                return False  # 停止监听，退出程序
        # 更新上一次按键和上一次回车时间
        last_key = "enter"
        last_enter_time = current_time

    else:
        # 非回车按键，重置上一次按键记录
        last_key = None
    
    # 3. 如需持续监听，返回 None（pynput 要求回调返回 None 继续监听）
    return None

def global_keyboard_listener():
    """
    启动全局键盘监听
    """
    global press_status
    print("开始全局监听键盘输入（连续两次按下回车返回 1，按 Ctrl+C 退出）...")
    try:
        # 启动监听器，持续监听键盘按下事件
        with Listener(on_press=on_press) as listener:
            listener.join()
        if press_status == 1:
            return 1
    except KeyboardInterrupt:
        print("\n程序被用户手动终止，退出全局监听")

if __name__ == "__main__":
    global_keyboard_listener()