from pywinauto import Application, timings
from pywinauto.findwindows import ElementNotFoundError

MAIN_WINDOW_TITLE = "Lidar ME"
WAIT_MAIN_WINDOW = 20
WAIT_DIALOG = 15
WAIT_ACTION = 5
APP_WORKDIR = None
APP_EXE = r"C:\Program Files\Lidar-ME\Lidar ME\Lidar ME.exe"

def start_or_connect_app() -> Application:
    try:
        app = Application(backend="uia").connect(title=MAIN_WINDOW_TITLE, timeout=5)
        return app
    except Exception:
        start_kwargs = {"timeout": WAIT_MAIN_WINDOW}
        if APP_WORKDIR:
            start_kwargs["work_dir"] = APP_WORKDIR
        app = Application(backend="uia").start(APP_EXE, **start_kwargs)
        return app

app = start_or_connect_app()
window = app.window(title=MAIN_WINDOW_TITLE)
window.wait("ready", timeout=WAIT_MAIN_WINDOW)
window.set_focus()
print(window.print_ctrl_ids())