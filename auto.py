import logging
import re
import time
import ctypes
import sys
import numpy as np
import cv2
import os

from PIL import Image
from ctypes import wintypes
#from curses.textpad import rectangle
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pywinauto.mouse
from PIL import ImageGrab
from pywinauto import Application, timings, mouse
from pywinauto.findwindows import ElementNotFoundError
from pywinauto.keyboard import send_keys
#from visual import get_board_centers_from_screen, solve_board_center_bbox_from_two_markers
from scantest import detect_target, match_target_by_template, warp_by_bbox
from keylistener import global_keyboard_listener
from freegrab import grab_quadrilateral, _order_points

ROOT_STR = r"C:\Users\ZN191014\Desktop\新建文件夹 (2)"
POINTCLOUD_ROOT = Path(ROOT_STR)
PROCESSED_MARK_DIR = POINTCLOUD_ROOT / "after"
APP_EXE = r"C:\Program Files\Lidar-ME\Lidar ME\Lidar ME.exe"
APP_WORKDIR = None
MAIN_WINDOW_TITLE = "Lidar ME"
NEW_TASK_BUTTON_TEXT = "新建任务"
SELECT_DIALOG_TITLE_RE = r".*选择文件夹.*"
SELECT_BUTTON_TEXT = "选择文件夹"
RTK_BUTTON_TEXT = "RTK校准"
CONFIRM_DIALOG_TITLE_RE = r".*确认.*"
CONFIRM_BUTTON_TEXT = "确认"
WAIT_MAIN_WINDOW = 20
WAIT_DIALOG = 15
WAIT_ACTION = 5
FOLDER_REGEX = re.compile(r"^\d{4}$")
L = 10 #标记宽度
sep = 5 #标记间缝隙宽度


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
    logging.FileHandler("process.log"), # 输出到文件
    logging.StreamHandler(sys.stdout) # 输出到控制台
    ])

def list_pending_folders(root: Path, marker_root: Path) -> Iterable[Path]:
    processed = set()
    if marker_root.exists():
        for item in marker_root.iterdir():
            if item.is_dir():
                base_name = item.name.rstrip("-")
                if FOLDER_REGEX.fullmatch(base_name):
                    processed.add(base_name)

    today = datetime.now().date()
    candidates = [
        candidate
        for candidate in root.iterdir()
        if candidate.is_dir() and FOLDER_REGEX.fullmatch(candidate.name)
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime)

    for folder in candidates:
        modified_date = datetime.fromtimestamp(folder.stat().st_mtime).date()
        if modified_date != today:
            continue
        if folder.name in processed:
            continue
        yield folder


def start_or_connect_app() -> Application:
    try:
        app = Application(backend="uia").connect(title=MAIN_WINDOW_TITLE, timeout=5)
        logging.info("Connected...")
        return app
    except Exception:
        logging.info("try to start application...")
        start_kwargs = {"timeout": WAIT_MAIN_WINDOW}
        if APP_WORKDIR:
            start_kwargs["work_dir"] = APP_WORKDIR
        app = Application(backend="uia").start(APP_EXE, **start_kwargs)
        return app


def ensure_main_window(app: Application):
    window = app.window(title=MAIN_WINDOW_TITLE)
    window.wait("ready", timeout=WAIT_MAIN_WINDOW)
    window.set_focus()
    #print(window.print_ctrl_ids())
    return window

# def get_ctrl_position(window,
#                       title:Optional[str]=None,
#                       control_type:Optional[str]=None,
#                       timeout:int=WAIT_ACTION,
#                       use_regex:bool=False) -> tuple[int,int]:
#     raw_title = title
#     pattern = None
#     if isinstance(title, re.Pattern):
#         pattern = title
#         title = None
#     elif title and (use_regex or re.search(r"[.^$*+?{}\[\]|()]", title)):
#         pattern = re.compile(title)
#         title = None
#
#     search_order = [control_type] if control_type else [None, "Button", "Text", "TabItem"]
#     last_exc = None
#
#     for ctrl_type in search_order:
#         try:
#             kwargs = {}
#             if ctrl_type:
#                 kwargs["control_type"] = ctrl_type
#             if pattern is not None:
#                 kwargs["title_re"] = pattern
#             elif title is not None:
#                 kwargs["title"] = title
#
#             ctrl = window.child_window(**kwargs)
#             ctrl.wait("enabled", timeout=timeout)
#             rect = ctrl.rectangle()
#             return (rect.left, rect.top)
#         except (ElementNotFoundError, timings.TimeoutError) as exc:
#             last_exc = exc

def click_button(
    window,
    title: Optional[str] = None,
    button_type: Optional[str] = None,
    timeout: int = WAIT_ACTION,
    use_regex: bool = False,
    click: bool = True,
    offset: Optional[tuple[int, int]] = None,
    stable: bool = True
):
    raw_title = title
    pattern = None
    if isinstance(title, re.Pattern):
        pattern = title
        title = None
    elif title and (use_regex or re.search(r"[.^$*+?{}\[\]|()]", title)):
        pattern = re.compile(title)
        title = None

    search_order = [button_type] if button_type else [None, "Button", "Text", "TabItem"]
    last_exc = None

    for ctrl_type in search_order:
        try:
            kwargs = {}
            if ctrl_type:
                kwargs["control_type"] = ctrl_type
            if pattern is not None:
                kwargs["title_re"] = pattern
            elif title is not None:
                kwargs["title"] = title

            if stable:
                btn = window.child_window(**kwargs)
            else:
                btn = window[title]
            btn.wait("enabled", timeout=timeout)
            pos = btn.rectangle()
            pos_dict = {}
            pos_dict["left"] = pos.left
            pos_dict["top"] = pos.top
            pos_dict["right"] = pos.right
            pos_dict["bottom"] = pos.bottom

            if offset:
                btn.move_mouse_input()
                pt = wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                print(pt.x, pt.y)
                target_pos = (pt.x + offset[0], pt.y + offset[1])
                mouse.move(coords=target_pos)
                if click: mouse.click(button="left", coords=target_pos)
            else:
                if click: btn.click_input()
            return btn, pos_dict
        except (ElementNotFoundError, timings.TimeoutError) as exc:
            last_exc = exc

    logging.error("NotFound: title=%s regex=%s type=%s", raw_title or (pattern.pattern if pattern else None), bool(pattern), button_type)
    #window.print_ctrl_ids()
    if last_exc:
        raise last_exc
    raise ElementNotFoundError("未找到匹配控件")


def wait_dialog(app: Application, title_re: str, timeout: int):
    timings.wait_until_passes(timeout, 0.5, lambda: app.window(title_re=title_re))
    dialog = app.window(title_re=title_re, top_level_only=True, visible_only=True)
    dialog.wait("ready", timeout=timeout)
    dialog.set_focus()
    return dialog

def escape_send_keys(text: str) -> str:
    return text.replace("(", "{(}").replace(")", "{)}").replace(" ", "{SPACE}")

def scroll_delay(center: tuple[int, int], wheel_dist: int=1, mouse: object=pywinauto.mouse):
    """封装pywinauto.mouse滚轮滚动，增加延时,防止多次快速滚动请求导致软件忽略后续滚动动作，wheel_dist 正数向上滚动，负数向下滚动"""
    for i in range(abs(wheel_dist)):
        if wheel_dist < 0:
            mouse.scroll(coords=center, wheel_dist=-1)
            time.sleep(0.2)
        elif wheel_dist > 0:
            mouse.scroll(coords=center, wheel_dist=1)
            time.sleep(0.2)
        else:
            break

def click_view_for_task(window, task_name, timeout=3, title="查看"):
    task_name = (task_name or "").strip()
    title = (title or "").strip()

    task_re = re.compile(rf"^{re.escape(task_name)}(\(\d+\))?$")
    today_time_re = re.compile(r"^今天\s*\d{{1,2}}:\d{{2}}$")

    def find_task_items():
        items = []
        try:
            for di in window.descendants(control_type="DataItem"):
                t = (di.window_text() or "").strip()
                if task_re.fullmatch(t):
                    items.append(di)
        except Exception:
            pass
        return items

    def row_has_today_time(row):
        try:
            for w in row.descendants():
                t = (w.window_text() or "").strip()
                if today_time_re.match(t):
                    return True
        except Exception:
            pass
        return False

    # 1) 等待匹配到任务行（可能有多个：task / task(1) / task(2)）
    end_t = time.time() + timeout
    candidates = []
    while time.time() < end_t:
        candidates = find_task_items()
        if candidates:
            break
        time.sleep(0.2)

    if not candidates:
        logging.error("Not Found taskline: %s", task_name)
        return None

    # 2) 若多个，优先选父容器里带“今天 HH:MM”的那一行
    if len(candidates) == 1:
        task_item = candidates[0]
    else:
        scored = []
        for di in candidates:
            row = di.parent()
            scored.append((row_has_today_time(row), di))

        # True 优先
        scored.sort(key=lambda x: x[0], reverse=True)
        task_item = scored[0][1]

    # 3) 取父容器（同一行/同一组的容器）
    row = task_item.parent()

    # 4) 找同一父容器下动作列（通常文本以“查看”开头：查看 目录 原文件 删除）
    try:
        actions_cell = row.child_window(title_re=rf"^{re.escape(title)}.*", control_type="DataItem")
        actions_cell.wait("exists enabled visible", timeout=timeout)
        return actions_cell
    except Exception:
        actions_w = None
        for di in row.descendants(control_type="DataItem"):
            t = (di.window_text() or "").strip()
            if t.startswith(title):
                actions_w = di
                break

        if actions_w is None:
            return None

        # 5) 在动作列里找真正可点击的 “查看”（可能是 Text / Static / Hyperlink / Button）
        for ct in ("Text", "Static", "Hyperlink", "Button"):
            for w in actions_w.descendants(control_type=ct):
                if (w.window_text() or "").strip() == title:
                    return w

        return None
        
def choose_folder(dialog, folder_name: str, parent: str=ROOT_STR) -> None:
    # address_edit = None
    # dialog.print_ctrl_ids()
    # for edit in dialog.descendants(control_type="Edit"):
    #     name = edit.element_info.name or ""
    #     if "地址" in name or "路径" in name or address_edit is None:
    #         address_edit = edit
    #         break
    #         # if "地址" in name or "路径" in name:
    #         #     break
    # if address_edit is None:
    #     logging.error("未找到地址栏控件，控件树如下：")
    #     dialog.print_ctrl_ids()
    #     raise ElementNotFoundError("未找到地址栏 Edit 控件") visual
    
    dialog.set_focus()
    # try:
    #     click_button(dialog,title="此电脑", button_type="TreeItem", timeout=2)
    # except timings.TimeoutError:
    #     # 防止选择“此电脑”失败（如菜单被滚动出屏幕外）
    #     for i in range(2):
    #         # 多点几次确保之后点击地址输入文字正常
    #         click_button(dialog,title="向上一级区段工具栏", button_type="ToolBar")
    click_button(dialog,title="^(地址|address|add):.*", button_type="ToolBar", use_regex=True, timeout=WAIT_MAIN_WINDOW)
    #click_button(dialog, title="上一个位置", button_type="Button", offset=(-2, 0))
    send_keys(escape_send_keys(parent) + "{\}" + folder_name + "{ENTER}")
    try:
        click_button(dialog,title="选择文件夹", button_type="Button")
    except pywinauto.findwindows.ElementAmbiguousError:
        click_button(dialog,title="选择文件夹", button_type="Button")
    # time.sleep(3)
    # click_button(window,title="坐标转换")
    # send_keys(str(parent) + "{ENTER}", with_spaces=True, pause=0.05)
    # time.sleep(0.5)
    # send_keys(folder_name, with_spaces=True, pause=0.05)
    # time.sleep(0.2)
    # send_keys("{ENTER}")
    # time.sleep(0.3)
    # click_button(dialog, SELECT_BUTTON_TEXT, timeout=WAIT_ACTION)

def save_clipboard_img_to_dir(folder_name: str, checktime: int=2) -> None:
    """
    阻塞等待剪贴板出现有效图片，然后保存到 PROCESSED_MARK_DIR/img 目录下，命名为 folder_name
    """
    # 1. 拼接目标目录路径（PROCESSED_MARK_DIR + img）
    target_dir = PROCESSED_MARK_DIR / "img"
    
    # 2. 创建目标目录（如果不存在），递归创建多级目录
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"目录不存在，已创建：{target_dir}")
    
    print(f"开始监听剪贴板，每 {checktime} 秒检测一次，检测到图片后自动保存（可按 Ctrl+C 退出）...")
    
    try:
        from PIL.Image import Image 
        # 3. 无限循环，阻塞等待剪贴板出现有效图片
        while True:
            # 从剪贴板获取内容
            clipboard_content = ImageGrab.grabclipboard()
            
            # 4. 判断是否为有效图片
            if clipboard_content is not None and isinstance(clipboard_content, Image):
                # 5. 拼接完整保存路径（添加 .png 后缀，确保图片可正常打开）
                img_save_path = target_dir / f"{folder_name}.png"
                
                # 6. 保存图片（quality=95 优化图片质量，可根据需求调整）
                clipboard_content.save(img_save_path, format="PNG", quality=95)
                print(f"\n图片保存成功！路径：{img_save_path}")
                break  # 保存成功，退出循环，结束阻塞
            else:
                # 无有效图片，延迟后继续检测
                print(f"当前剪贴板无有效图片，{checktime} 秒后重新检测...", end="\r")
                time.sleep(checktime)
    
    except KeyboardInterrupt:
        # 捕获 Ctrl+C，实现手动退出
        print("\n\n程序被用户手动终止，退出监听")
    except Exception as e:
        print(f"\n程序运行出错：{str(e)}")

def pil_save_cv(img: np.ndarray, save_path: str | Path, *, quality=95) -> None:
    save_path = str(save_path)

    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("img 为空或不是有效的 OpenCV ndarray 图像")

    # 确保目录存在
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # 重名规则：同名直接替换（先删掉旧文件更稳）
    if os.path.exists(save_path):
        try:
            os.remove(save_path)
        except Exception:
            # 例如只读/占用：尽量改权限再删
            try:
                os.chmod(save_path, 0o666)# 修改为可读写
                os.remove(save_path)
            except Exception:
                pass

    if img.ndim == 2:
        pil_img = Image.fromarray(img)  # 灰度
    elif img.ndim == 3 and img.shape[2] == 3:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif img.ndim == 3 and img.shape[2] == 4:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    else:
        raise ValueError(f"不支持的图像形状: {img.shape}")

    # PNG 不看 quality；JPEG 才看 quality
    pil_img.save(save_path, quality=quality)

def notify_start_select_points(timeout_ms: int = 1000, title: str = "提示", hwnd=None, msg: str = "可以开始选点了"):
    """Windows 简单弹窗提示，timeout_ms 毫秒后自动关闭；失败则降级为 print。"""
    try:
        owner = int(hwnd) if hwnd else 0
        # 0 = MB_OK
        ctypes.windll.user32.MessageBoxTimeoutW(
            0,
            msg,
            title,
            0,
            0,
            int(timeout_ms),
        )
    except Exception:
        print(f"{title}: {msg}")

def solve(window, app: Application, folder_name: str, button=None):
    """解算过程, return: 0失败，没找到坐标转换按钮，要求重试"""

    # 定义要是别的板子名字和板子含有的charuco id
    # target_ids = [392, 393, 394, 395]
    # groups = {
    #     "boardA_392_393": [target_ids[0],target_ids[1]],
    #     "boardB_394_395": [target_ids[2],target_ids[3]],
    # }

    # # 坐标转换
    while True:
        try:
            click_button(window, title="坐标转换", button_type="Text", timeout=5)
            break
        except timings.TimeoutError:
            #button.click_input()
            time.sleep(1)
            return 0
    while True:
        try:
            click_button(window, title="请选择", button_type="Text")
            break
        except ElementNotFoundError:
            time.sleep(0.5)
    send_keys("{DOWN}{ENTER}")
    click_button(window, title="请选择", button_type="Button")
    # 选择文件夹的子窗口
    time.sleep(0.5)
    dialog = wait_dialog(app, SELECT_DIALOG_TITLE_RE, WAIT_DIALOG)
    choose_folder(dialog,"after"+"{\}"+ "csv" + "{\}" + folder_name + "{-}")
    click_button(window, title="确认", button_type="Button")

    while True:
        try:
            click_button(window, title="浏览", button_type="Button",timeout=5)
            break
        except ElementNotFoundError:
            time.sleep(0.5)
        except timings.TimeoutError:
            logging.error("可能是点云未固定")
            click_button(window, title="返回", button_type="Button")
            click_button(window, title="主页", button_type="TabItem")
            return -1
    time.sleep(6)

    while True:
        try:
            click_button(window,title="处理中",click=False, timeout=1)
            time.sleep(0.5)
        except (ElementNotFoundError, timings.TimeoutError):
            print("处理完成")
            break

    # 截图，手动模式下注释掉L33-373
    # click_button(window,title="俯视")
    # time.sleep(0.3)
    # point_cloud_image = click_button(window,title="窗口Image2",click=False,stable=False)
    # pcp, pcp_loc = point_cloud_image
    # center = (int((pcp_loc["left"] + pcp_loc["right"])*(11/20)), int((pcp_loc["bottom"] + pcp_loc["top"])* (12/20)))
    # scroll_delay(center=center, wheel_dist= 11, mouse=mouse)
    # centers, records, image, detection = get_board_centers_from_screen(groups=groups)
    # corners_dict = {}
    # for mid in target_ids:
    #     pts = records.get(mid, [])
    #     if len(pts) == 2:
    #         corners_dict[mid] = np.array(pts, dtype=np.float32).reshape(1, 4, 2)
    #     else:
    #         logging.warning("未检测到完整标记 ID=%d 的角点，跳过。", mid)
    # newcenter1, bbox1 = solve_
    # 
    # 
    # board_center_bbox_from_two_markers(image, corners_dict, (392, 393), L, sep)
    # newcenter2, bbox2 = solve_board_center_bbox_from_two_markers(image, corners_dict, (394, 395), L, sep)
    # time.sleep(0.5)
    logging.debug("Start detect target...")
    point_cloud_image = click_button(window,title="窗口Image2",click=False,stable=False)
    pcp, pcp_loc = point_cloud_image
    center = (int((pcp_loc["left"] + pcp_loc["right"])*(11/20)), int((pcp_loc["bottom"] + pcp_loc["top"])* (13/20)))
    scroll_delay(center=center, wheel_dist= 10, mouse=mouse)
    detected_img, bbox = detect_target(mode=3, visualize=False)

    if bbox is not None:
        # 截取标靶四边形区域
        img1 = warp_by_bbox(detected_img, bbox, (800, 800))
        img_match = match_target_by_template("image.png", img1, mode=1, visualize=False)
        for i in img_match:
            if i is None:
                logging.error("failed to match target position, please select point manually")
                print("标靶位置检测失败，请手动确认截图选点")
                img1 = detected_img
                break
    # img1 = ImageGrab.grab(bbox=(bbox1[1], bbox1[0], bbox1[3], bbox1[2]))
    img1_save_path = PROCESSED_MARK_DIR / f"img\\{folder_name}.png"
    try:
        img1.save(f"{ROOT_STR}\\after\\img\\{folder_name}.png")
    except AttributeError:
        # img1 是 ndarray
        pil_save_cv(img1, img1_save_path)

    # img2 = ImageGrab.grab(bbox=(bbox2[1], bbox2[0], bbox2[3], bbox2[2]))
    # cv2.imwrite(f"{ROOT_STR}\\after\\img\\{folder_name}marker2.png", cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR))
    # #选点
    # try:
    #     click_button(window, title="真彩", click=False)
    # except ElementNotFoundError:
    #     try:
    #         click_button(window, title="高程", click=False)
    #     except ElementNotFoundError:
    #         pass
    # click_button(window, title="强度")
    # click_button(window, title="坐标")

    # if newcenter1 is not None:
    #     cx, cy = newcenter1
    #     mouse.click(button="left", coords=(int(cx), int(cy)))
    # if newcenter2 is not None:
    #     cx, cy = newcenter2
    #     mouse.click(button="left", coords=(int(cx), int(cy)))

    # 将剪切板的图片保存为文件
    #save_clipboard_img_to_dir(folder_name)
    # 手动选点
    # 系统弹窗提醒可以开始选点
    notify_start_select_points(timeout_ms=1000, title="选点提示", hwnd=window.handle)
    for i in range(5):
        try:
            click_button(window, title="真彩", button_type="Text", click=True)
            break
        except ElementNotFoundError:
            try:
                click_button(window, title="高程", button_type="Text", click=True)
                break
            except ElementNotFoundError:
                pass
    click_button(window, title="强度")
    click_button(window, title="坐标", button_type="Text", click=True)

    # 等待用户按下两次回车
    print("请在软件界面手动选点，选点完成后连续按两次回车以继续...")
    while global_keyboard_listener() != 1:
        time.sleep(0.5)
    print("检测到连续两次回车，继续执行后续操作...")
    time.sleep(0.5)
    # 导出坐标
    while True:
        try:
            click_button(window, title="导出坐标", button_type="Text")
            break
        except ElementNotFoundError or timings.TimeoutError:
            logging.error("Click Error: Retrying...")
            time.sleep(1)
    time.sleep(0.5)
    click_button(window, title="导出", button_type="Button")
    dialog = wait_dialog(app, SELECT_DIALOG_TITLE_RE, WAIT_DIALOG)
    for i in range(3):
        try:
            click_button(dialog, title="选择文件夹", button_type="Button")
            break
        except timings.TimeoutError:
            time.sleep(0.3)


def run_import_for_folder(app: Application, folders_list:list) -> None:
    find = False #是否找到任务行
    window = ensure_main_window(app)
    click_button(window, title="主页", button_type="TabItem")
    # for folder in folders_list:
    #     logging.info("开始导入: %s", folder)
    #     click_button(window, NEW_TASK_BUTTON_TEXT, "Text", 2)
    #     #time.sleep(WAIT_ACTION)
    #     click_button(window, "选择文件夹", "Button", 2)
    #     dialog = wait_dialog(app, SELECT_DIALOG_TITLE_RE, WAIT_DIALOG)
    #     choose_folder(dialog, folder.name, folder.parent.as_posix())
    #     click_button(window,title="RTK校准")
    #     click_button(window,title="确认")
    #     time.sleep(3.0)
    #     window = ensure_main_window(app)
    #     logging.info("导入完成: %s", folder)
    #     time.sleep(0.2)

    # time.sleep(1.0)
    logging.info("Finished importing all folders, start processing...")


    for folder in folders_list:
        #阻塞直到导入完成
        #c = click_view_for_task(window, folder.name)
        ret = 0
        logging.error("Cannot find addr connvert button, retrying...")
        click_button(window, "1", button_type="Text", timeout=2)
        #c = click_view_for_task(window, folder.name)
        for i in range(8): #最多翻8页
            c = click_view_for_task(window, folder.name)
            if c is not None:
                c.click_input()
                find = True
                break
            else:
                click_button(window, "下一页",button_type="Button", timeout=2)

        if not find:
            logging.error("Cannot find task row: %s, skipping this folder and retrying.", folder.stem)
            continue
        time.sleep(2)#等待界面转换
        #click_view_for_task(window, "")#测试用，点开别的任务
        ret = solve(window, app, folder.stem)
        if ret == -1:
            logging.error("Coordinate conversion failed, skipping this folder %s retry", folder.stem)
            continue
        time.sleep(1)
        for i in range(3):
            click_button(window, title="主页", button_type="TabItem", timeout=2)
            time.sleep(0.3)
            
        #solve(window, app, folder.stem)

        # 将points.csv移动到after/csv目录下，并改名为对应{folder.stem}.csv ，这一部分（~508）已经更改路径结构，不需要再移动了
        # src_csv = Path(folder) / "points.csv"
        # dest_dir = POINTCLOUD_ROOT / "after" / "csv"
        # dest_dir.mkdir(parents=True, exist_ok=True)
        # dest_csv = dest_dir / f"{folder.stem}.csv"
        # if src_csv.exists():
        #     # rename 相当于移动文件和重命名
        #     src_csv.rename(dest_csv)
        #     logging.info("已移动并重命名坐标文件到: %s", dest_csv)
        # else:
        #     logging.warning("未找到坐标文件: %s", src_csv)

    # try:
    #     confirm = wait_dialog(app, CONFIRM_DIALOG_TITLE_RE, WAIT_DIALOG)
    #     click_button(confirm, CONFIRM_BUTTON_TEXT)
    # except ElementNotFoundError:
    #     logging.warning("未见确认对话框，继续。")



def main() -> None:
    setup_logging()
    logging.info("Start...")
    # 一次只能输入10个文件夹，防止软件无法判断是结算没完成还是在下一页
    #pending = list(list_pending_folders(POINTCLOUD_ROOT, PROCESSED_MARK_DIR))
    pending = []
    # for test only
    pending.append(Path(ROOT_STR + "\\" + "0973"))
    pending.append(Path(ROOT_STR + "\\" + "1005"))
    # if not pending:
    #     logging.info("没有待导入的四位数点云文件夹。")
    #     return
    app = start_or_connect_app()
    try:
        run_import_for_folder(app, pending)
        time.sleep(2)
    except Exception as exc:
        logging.exception("Process %s Error: %s", pending, exc)
    logging.info("All done...")



if __name__ == "__main__":
    main()