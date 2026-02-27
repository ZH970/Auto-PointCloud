import os
import sys
import json
import queue
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

WORKDIR = Path(__file__).resolve().parent
CONFIG_PATH = WORKDIR / "ui_config.json"

ENV_KEYS = {
    # main.py
    "SPC_SRC_LABEL": "LidarME",
    "SPC_SRC_SUBPATH": "",
    "SPC_DST": r"C:\Users\ZN191014\Desktop\新建文件夹 (2)",
    "SPC_MIN_MB": "290",
    "SPC_MAX_MB": "680",
    "SPC_REQUIRED_MOD_TIME": "2",
    "SPC_REQUIRED_MOD_DATE": "2026-01-29",
    "SPC_DRY_RUN": "0",
    "SPC_VERBOSE": "1",
    # auto.py
    "SPC_ROOT_STR": r"C:\Users\ZN191014\Desktop\新建文件夹 (2)",
    "SPC_APP_EXE": r"C:\Program Files\Lidar-ME\Lidar ME\Lidar ME.exe",
    "SPC_PENDING": "",  # 例如 "1172,1173"；留空则按 auto.py 自己的 list_pending_folders 规则
    # points2SLAM_Output.py
    "SPC_CSV": "",
    "SPC_WORKBOOK": "",
    "SPC_SHEET": "SLAM扫描测试",
    "SPC_SAVE_AS": "",
    "SPC_PRESERVE_IMAGE_COLUMN": "1",
}

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PointCloud 工具箱")
        self.geometry("980x680")

        self.log_q: "queue.Queue[str]" = queue.Queue()

        cfg = _load_config()
        self.vars = {k: tk.StringVar(value=str(cfg.get(k, ENV_KEYS[k]))) for k in ENV_KEYS}

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        frm1 = ttk.Frame(nb)
        frm2 = ttk.Frame(nb)
        frm3 = ttk.Frame(nb)
        nb.add(frm1, text="1) TF卡导入 (main.py)")
        nb.add(frm2, text="2) LidarME 自动化 (auto.py)")
        nb.add(frm3, text="3) 写入Excel (points2SLAM_Output.py)")

        self._build_main_tab(frm1)
        self._build_auto_tab(frm2)
        self._build_excel_tab(frm3)

        # 底部日志
        bottom = ttk.Frame(self)
        bottom.pack(fill="both", expand=False)
        ttk.Label(bottom, text="输出日志：").pack(anchor="w", padx=8, pady=(6, 0))

        self.txt = tk.Text(bottom, height=14, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=8, pady=8)

        self.after(100, self._drain_logs)

        # 顶部按钮
        top = ttk.Frame(self)
        top.pack(fill="x", side="bottom")
        ttk.Button(top, text="保存配置", command=self.save_cfg).pack(side="right", padx=8, pady=6)

    def _row(self, parent, r, label, key, browse: bool = False, kind: str = "dir"):
        ttk.Label(parent, text=label, width=22).grid(row=r, column=0, sticky="w", padx=8, pady=6)
        ent = ttk.Entry(parent, textvariable=self.vars[key])
        ent.grid(row=r, column=1, sticky="we", padx=8, pady=6)
        if browse:
            def _pick():
                if kind == "dir":
                    p = filedialog.askdirectory()
                else:
                    p = filedialog.askopenfilename()
                if p:
                    self.vars[key].set(p)
            ttk.Button(parent, text="选择", command=_pick).grid(row=r, column=2, padx=8, pady=6)
        parent.columnconfigure(1, weight=1)

    def _build_main_tab(self, parent):
        self._row(parent, 0, "卷标 SRC_LABEL", "SPC_SRC_LABEL")
        self._row(parent, 1, "子路径 SRC_SUBPATH", "SPC_SRC_SUBPATH")
        self._row(parent, 2, "目标目录 DST", "SPC_DST", browse=True, kind="dir")
        self._row(parent, 3, "MIN_MB", "SPC_MIN_MB")
        self._row(parent, 4, "MAX_MB", "SPC_MAX_MB")
        self._row(parent, 5, "REQUIRED_MOD_TIME(0/1/2)", "SPC_REQUIRED_MOD_TIME")
        self._row(parent, 6, "REQUIRED_MOD_DATE(YYYY-MM-DD)", "SPC_REQUIRED_MOD_DATE")
        self._row(parent, 7, "DRY_RUN(0/1)", "SPC_DRY_RUN")
        self._row(parent, 8, "VERBOSE(0/1)", "SPC_VERBOSE")

        ttk.Button(parent, text="运行 main.py（导入/重命名）", command=self.run_main).grid(
            row=9, column=0, columnspan=3, sticky="we", padx=8, pady=(14, 10)
        )

    def _build_auto_tab(self, parent):
        self._row(parent, 0, "点云根目录 ROOT_STR", "SPC_ROOT_STR", browse=True, kind="dir")
        self._row(parent, 1, "LidarME 程序路径 APP_EXE", "SPC_APP_EXE", browse=True, kind="file")
        self._row(parent, 2, "指定待处理(逗号分隔)", "SPC_PENDING")

        ttk.Button(parent, text="运行 auto.py（自动操作）", command=self.run_auto).grid(
            row=3, column=0, columnspan=3, sticky="we", padx=8, pady=(14, 10)
        )

    def _build_excel_tab(self, parent):
        self._row(parent, 0, "points.csv 路径", "SPC_CSV", browse=True, kind="file")
        self._row(parent, 1, "Excel 工作簿路径", "SPC_WORKBOOK", browse=True, kind="file")
        self._row(parent, 2, "Sheet 名称", "SPC_SHEET")
        self._row(parent, 3, "另存为(可空)", "SPC_SAVE_AS", browse=True, kind="file")
        self._row(parent, 4, "保留图片列(默认1)", "SPC_PRESERVE_IMAGE_COLUMN")

        ttk.Button(parent, text="运行 points2SLAM_Output.py（写入Excel）", command=self.run_excel).grid(
            row=5, column=0, columnspan=3, sticky="we", padx=8, pady=(14, 10)
        )

    def save_cfg(self):
        cfg = {k: self.vars[k].get() for k in self.vars}
        _save_config(cfg)
        self._log(f"[UI] 配置已保存：{CONFIG_PATH}")

    def _build_env(self) -> dict:
        env = os.environ.copy()
        for k, v in self.vars.items():
            env[k] = v.get()
        return env

    def _run_script(self, script: str, args: list[str]):
        env = self._build_env()
        py = sys.executable
        cmd = [py, str(WORKDIR / script)] + args

        self._log(f"\n$ {' '.join(cmd)}\n")
        try:
            p = subprocess.Popen(
                cmd,
                cwd=str(WORKDIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert p.stdout is not None
            for line in p.stdout:
                self._log(line.rstrip("\n"))
            rc = p.wait()
            self._log(f"[DONE] {script} exit={rc}")
        except Exception as e:
            self._log(f"[ERROR] 运行失败：{e}")

    def _run_bg(self, func):
        t = threading.Thread(target=func, daemon=True)
        t.start()

    def run_main(self):
        self.save_cfg()
        self._run_bg(lambda: self._run_script("main.py", []))

    def run_auto(self):
        self.save_cfg()
        self._run_bg(lambda: self._run_script("auto.py", []))

    def run_excel(self):
        self.save_cfg()
        args = []
        csv_path = self.vars["SPC_CSV"].get().strip()
        wb_path = self.vars["SPC_WORKBOOK"].get().strip()
        sheet = self.vars["SPC_SHEET"].get().strip()
        save_as = self.vars["SPC_SAVE_AS"].get().strip()
        pic_col = self.vars["SPC_PRESERVE_IMAGE_COLUMN"].get().strip()

        if csv_path:
            args += ["--csv", csv_path]
        if wb_path:
            args += ["--workbook", wb_path]
        if sheet:
            args += ["--sheet", sheet]
        if save_as:
            args += ["--save-as", save_as]
        if pic_col:
            args += ["--preserve-image-column", pic_col]

        self._run_bg(lambda: self._run_script("points2SLAM_Output.py", args))

    def _log(self, msg: str):
        self.log_q.put(msg)

    def _drain_logs(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.txt.insert("end", msg + "\n")
                self.txt.see("end")
        except queue.Empty:
            pass
        self.after(100, self._drain_logs)

if __name__ == "__main__":
    App().mainloop()