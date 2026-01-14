import logging
import re
import time
import ctypes
from ctypes import wintypes
#from curses.textpad import rectangle
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pywinauto.mouse
from pywinauto import Application, timings, mouse
from pywinauto.findwindows import ElementNotFoundError
from pywinauto.keyboard import send_keys
from visual import get_board_centers_from_screen

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


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def list_pending_folders(root: Path, marker_root: Path) -> Iterable[Path]:
    # processed = set()
    # if marker_root.exists():
    #     for item in marker_root.iterdir():
    #         if item.is_dir():
    #             base_name = item.name.rstrip("-")
    #             if FOLDER_REGEX.fullmatch(base_name):
    #                 processed.add(base_name)

    today = datetime.now().date()
    candidates = [
        candidate
        for candidate in root.iterdir()
        if candidate.is_dir() and FOLDER_REGEX.fullmatch(candidate.name)
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime)

    for folder in candidates:
        modified_date = datetime.fromtimestamp(folder.stat().st_mtime).date()
        # if modified_date != today:
        #     continue
        yield folder
        # if folder.name not in processed:
        #     yield folder


def start_or_connect_app() -> Application:
    try:
        app = Application(backend="uia").connect(title=MAIN_WINDOW_TITLE, timeout=5)
        logging.info("已连接到正在运行的程序。")
        return app
    except Exception:
        logging.info("尝试启动程序。")
        start_kwargs = {"timeout": WAIT_MAIN_WINDOW}
        if APP_WORKDIR:
            start_kwargs["work_dir"] = APP_WORKDIR
        app = Application(backend="uia").start(APP_EXE, **start_kwargs)
        return app


def ensure_main_window(app: Application):
    window = app.window(title=MAIN_WINDOW_TITLE)
    window.wait("ready", timeout=WAIT_MAIN_WINDOW)
    window.set_focus()
    # print(window.print_ctrl_ids())
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

    logging.error("未找到控件: title=%s regex=%s type=%s", raw_title or (pattern.pattern if pattern else None), bool(pattern), button_type)
    # window.print_ctrl_ids()
    if last_exc:
        raise last_exc
    raise ElementNotFoundError("未找到匹配控件")


def wait_dialog(app: Application, title_re: str, timeout: int):
    timings.wait_until_passes(timeout, 0.5, lambda: app.window(title_re=title_re))
    dialog = app.window(title_re=title_re)
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
    click_button(dialog,title="此电脑", button_type="TreeItem")
    click_button(dialog,title="^(地址|address|add):.*", button_type="ToolBar", use_regex=True, timeout=WAIT_MAIN_WINDOW)
    #click_button(dialog, title="上一个位置", button_type="Button", offset=(-2, 0))
    send_keys(escape_send_keys(parent) + "{\}" + folder_name + "{ENTER}")
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

def solve(window, app: Application, folder_name: str):
    """解算过程"""

    # 定义要是别的板子名字和板子含有的charuco id
    groups = {
        "boardA_392_393": [392, 393],
        "boardB_394_395": [394, 395],
    }

    # 坐标转换
    click_button(window, title="坐标转换")
    click_button(window, title="请选择", button_type="Text")
    send_keys("{DOWN}{ENTER}")
    click_button(window, title="请选择", button_type="Button")
    # 选择文件夹的子窗口
    time.sleep(0.5)
    dialog = wait_dialog(app, SELECT_DIALOG_TITLE_RE, WAIT_DIALOG)
    choose_folder(dialog,"after"+"{\}"+ folder_name + "{-}")
    click_button(window, title="返回")
    time.sleep(0.8)

    # 截图
    point_cloud_image = click_button(window,title="窗口Image2",click=False,stable=False)
    pcp, pcp_loc = point_cloud_image
    center = (int((pcp_loc["left"] + pcp_loc["right"])*(1/2)), int((pcp_loc["bottom"] + pcp_loc["top"])* (7/12)))
    scroll_delay(center=center, wheel_dist= 10, mouse=mouse)
    centers, records, detection = get_board_centers_from_screen(groups=groups)
    time.sleep(0.5)

    #选点
    click_button(window, title="真彩")
    click_button(window, title="强度")
    click_button(window, title="坐标")






def run_import_for_folder(app: Application, folder: Path) -> None:
    logging.info("开始导入: %s", folder)
    window = ensure_main_window(app)
    # click_button(window, NEW_TASK_BUTTON_TEXT, "Text", 2)
    # #time.sleep(WAIT_ACTION)
    # click_button(window, "选择文件夹", "Button", 2)
    # dialog = wait_dialog(app, SELECT_DIALOG_TITLE_RE, WAIT_DIALOG)
    # choose_folder(dialog, folder.name, folder.parent.as_posix())
    # click_button(window,title="RTK校准")
    # click_button(window,title="确认")
    # time.sleep(3.0)
    # window = ensure_main_window(app)

    solve(window, app, folder.stem)
    click_button(window, RTK_BUTTON_TEXT)
    try:
        confirm = wait_dialog(app, CONFIRM_DIALOG_TITLE_RE, WAIT_DIALOG)
        click_button(confirm, CONFIRM_BUTTON_TEXT)
    except ElementNotFoundError:
        logging.warning("未见确认对话框，继续。")
    time.sleep(1.0)
    logging.info("导入完成: %s", folder)


def main() -> None:
    setup_logging()
    pending = list(list_pending_folders(POINTCLOUD_ROOT, PROCESSED_MARK_DIR))
    if not pending:
        logging.info("没有待导入的四位数点云文件夹。")
        return
    app = start_or_connect_app()
    for folder in pending:
        try:
            run_import_for_folder(app, folder)
            time.sleep(2)
        except Exception as exc:
            logging.exception("处理 %s 时出现异常: %s", folder, exc)
            break


if __name__ == "__main__":
    main()