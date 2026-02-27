import os
import re
import sys
import shutil
import ctypes
import string
import subprocess
from datetime import datetime, timedelta
# 尝试导入 tkinter，用于弹窗重命名；若不可用则在运行时跳过弹窗
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
except Exception:
    tk = None
    simpledialog = None
    messagebox = None

# 配置（在此直接写定）
SRC_LABEL = "LidarME"               # 设备在资源管理器中显示的卷标，例如 LidarME
SRC_SUBPATH = r""                   # 在该盘下的子路径（不以盘符开头），如需要检查根目录则留空
DST = r"C:\Users\ZN191014\Desktop\新建文件夹 (2)"            # <- 这里改成你的目标目录
MIN_MB = 290.0
MAX_MB = 680.0
DRY_RUN = False                     # True 表示只列出不执行复制
VERBOSE = True                      # True 打印详细信息
REQUIRED_MOD_TIME = 2          # 1表示允许按修改日期过滤，且手动输入日期，2是只匹配最近7天,0表示不启用
REQUIRED_MOD_DATE = "2026-01-29"   # 格式：YYYY-MM-DD

def _apply_env_overrides():
    """允许 ui.py 通过环境变量覆盖 main.py 的配置（不破坏原有默认值）。"""
    global SRC_LABEL, SRC_SUBPATH, DST, MIN_MB, MAX_MB, DRY_RUN, VERBOSE, REQUIRED_MOD_TIME, REQUIRED_MOD_DATE

    SRC_LABEL = os.getenv("SPC_SRC_LABEL", SRC_LABEL)
    SRC_SUBPATH = os.getenv("SPC_SRC_SUBPATH", SRC_SUBPATH)
    DST = os.getenv("SPC_DST", DST)

    def _f(name, cur):
        v = os.getenv(name, "")
        try:
            return float(v) if v != "" else cur
        except Exception:
            return cur

    def _i(name, cur):
        v = os.getenv(name, "")
        try:
            return int(v) if v != "" else cur
        except Exception:
            return cur

    def _b(name, cur):
        v = os.getenv(name, "")
        if v == "":
            return cur
        return v.strip() not in ("0", "false", "False", "no", "NO")

    MIN_MB = _f("SPC_MIN_MB", MIN_MB)
    MAX_MB = _f("SPC_MAX_MB", MAX_MB)
    REQUIRED_MOD_TIME = _i("SPC_REQUIRED_MOD_TIME", REQUIRED_MOD_TIME)
    REQUIRED_MOD_DATE = os.getenv("SPC_REQUIRED_MOD_DATE", REQUIRED_MOD_DATE)
    DRY_RUN = _b("SPC_DRY_RUN", DRY_RUN)
    VERBOSE = _b("SPC_VERBOSE", VERBOSE)

def is_timestamp_dir(name):
    # 匹配 2025-11-11_15-50-12 这样的格式
    return re.match(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$', name) is not None

def get_folder_size_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                # 忽略不可访问的文件
                pass
    return total

def ensure_unique_dst(dst_parent, name):
    candidate = os.path.join(dst_parent, name)
    if not os.path.exists(candidate):
        return candidate

    suffix = 1
    while True:
        # 重名时：把旧目录改名为 *_old1, *_old2 ...
        old_name = f"{name}_old{suffix}"
        old_path = os.path.join(dst_parent, old_name)
        if not os.path.exists(old_path):
            os.rename(candidate, old_path)
            return candidate
        suffix += 1

def find_timestamp_dirs(src, depth=1):
    """
    返回符合命名规则的目录绝对路径列表（只检查一级子目录）。
    depth 参数保留但在本脚本中使用值 1。
    """
    matches = []
    src = os.path.abspath(src)
    if depth == 1:
        try:
            for name in os.listdir(src):
                p = os.path.join(src, name)
                if os.path.isdir(p) and is_timestamp_dir(name):
                    matches.append(p)
        except OSError:
            pass
        return matches

    # depth == 0 或 depth > 1: 使用 os.walk 并控制深度
    base_level = src.count(os.sep)
    for root, dirs, files in os.walk(src):
        level = root.count(os.sep) - base_level + 1  # +1 because immediate children are level 1
        # 如果 depth>0 且当前 level 超过深度，则跳过更深层
        if depth > 0 and level > depth:
            # 防止 os.walk 进入更深层（修改 dirs）
            dirs[:] = []
            continue
        for d in list(dirs):
            if is_timestamp_dir(d):
                matches.append(os.path.join(root, d))
    return matches

def bytes_to_mb(b):
    return b / (1024 * 1024)

def find_drive_root_by_label(label):
    """
    在 Windows 中遍历 A:..Z:，通过 GetVolumeInformationW 比较卷标，返回盘根如 'H:\\'。
    找不到返回 None。
    """
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if not os.path.exists(root):
            continue
        vol_name_buf = ctypes.create_unicode_buffer(1024)
        fs_name_buf = ctypes.create_unicode_buffer(1024)
        serial_number = ctypes.c_uint()
        max_component_len = ctypes.c_uint()
        file_system_flags = ctypes.c_uint()
        try:
            rc = ctypes.windll.kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(root),
                vol_name_buf,
                ctypes.sizeof(vol_name_buf),
                ctypes.byref(serial_number),
                ctypes.byref(max_component_len),
                ctypes.byref(file_system_flags),
                fs_name_buf,
                ctypes.sizeof(fs_name_buf),
            )
        except Exception:
            rc = 0
        if rc:
            vol = vol_name_buf.value
            if vol == label:
                return root
    return None

# 新增：清理用户输入的文件夹名（移除 Windows 非法字符并去掉尾部空格/点）
def sanitize_name(name):
    forbidden = '<>:"/\\|?*'
    cleaned = ''.join(c for c in name if c not in forbidden)
    cleaned = cleaned.rstrip(' .')
    return cleaned

# 新增：弹窗询问重命名（留空并确定或取消都表示不重命名），找不到 tkinter 则返回 None
def prompt_for_rename(current_name):
    if not simpledialog or not tk:
        if VERBOSE:
            print("无法弹出重命名窗口（tkinter 不可用），跳过重命名。")
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        res = simpledialog.askstring("重命名", f"当前目录名: {current_name}\n输入新的文件夹名（留空并确认表示不重命名）：", parent=root)
        root.destroy()
        if res is None:
            return None
        res = res.strip()
        if res == "":
            return None
        res = sanitize_name(res)
        return res if res else None
    except Exception:
        if VERBOSE:
            print("弹窗重命名时发生错误，跳过重命名。")
        return None

# 新增：从目录名解析时间（返回 datetime 或 None）
def parse_timestamp_from_name(name):
    try:
        return datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None

# 新增：优先使用 robocopy（Windows 本地复制，行为接近资源管理器），不可用时回退到 shutil.copytree
def copy_folder_explorer_like(src, dst):
    """
    复制 src 到 dst（dst 为目标目录名，若不存在则创建）。
    优先使用 robocopy（Windows），返回 True 表示复制成功。
    在非 Windows 或 robocopy 失败时尝试回退到 shutil.copytree（可能受策略影响）。
    """
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    # 确保父目录存在
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    # 在 Windows 上尝试使用 robocopy：创建目标目录然后将 src 内容复制到 dst
    if os.name == "nt":
        try:
            os.makedirs(dst, exist_ok=True)
            # robocopy: source_dir  destination_dir  [files]  /E 复制子目录包括空目录  /COPYALL 尽量保留属性
            cmd = ["robocopy", src, dst, "/E", "/COPYALL", "/R:1", "/W:1", "/NFL", "/NDL"]
            if VERBOSE:
                print("使用 robocopy 执行复制:", " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            rc = proc.returncode
            if VERBOSE:
                print("robocopy 返回码:", rc)
                if proc.stdout:
                    print("robocopy stdout:", proc.stdout)
                if proc.stderr:
                    print("robocopy stderr:", proc.stderr)
            # robocopy 返回码位域：0-7 为成功/警告， >=8 为失败
            if rc < 8:
                return True
            else:
                raise RuntimeError(f"robocopy 失败，返回码 {rc}")
        except FileNotFoundError:
            # robocopy 未找到（极少见），回退
            if VERBOSE:
                print("未找到 robocopy，回退到 shutil.copytree。")
        except Exception as e:
            if VERBOSE:
                print("调用 robocopy 出现异常，回退到 shutil.copytree。异常:", e)
    # 回退：使用 shutil.copytree（注意 Python 版本差异）
    try:
        # 如果目标目录不存在，直接 copytree；若已存在则复制内容到目标（Python 3.8+ 支持 dirs_exist_ok）
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
        else:
            # 将 src 内容复制到已有 dst
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
        return True
    except Exception as e:
        if VERBOSE:
            print("shutil.copytree 回退也失败，错误:", e)
        return False
    
def parse_args():
    # 解析源盘（通过卷标）
    resolved_root = find_drive_root_by_label(SRC_LABEL)
    if resolved_root:
        src = os.path.join(resolved_root, SRC_SUBPATH) if SRC_SUBPATH else resolved_root
    else:
        print(f"未在系统中找到标识为 '{SRC_LABEL}' 的盘。请确认设备已连接，或修改脚本中的 SRC_LABEL。")
        sys.exit(1)
    return src

def validate_paths(src, dst):
    if not os.path.isdir(src):
        print(f"源目录不存在或不是目录: {src}")
        sys.exit(1)
        return False
    if not os.path.exists(dst):
        try:
            os.makedirs(dst, exist_ok=True)
            return True
        except OSError as e:
            print(f"无法创建目标目录 {dst}: {e}")
            sys.exit(1)
            return False
    return True
    
def prompt_for_mod_date_filter(default_enabled: bool, default_date: str):
    """
    弹窗：勾选是否启用“按修改日期过滤”，并输入日期 YYYY-MM-DD
    返回: (enabled: bool, date_str: str|None)
    - 取消/关闭窗口 => (False, None)
    - 未勾选 => (False, None)
    """
    if not tk:
        return default_enabled, (default_date if default_enabled else None)

    date_default = (default_date or datetime.now().strftime("%Y-%m-%d")).strip()

    root = tk.Tk()
    root.title("按修改日期过滤")
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    enabled_var = tk.BooleanVar(value=bool(default_enabled))
    date_var = tk.StringVar(value=date_default)

    tk.Label(root, text="是否启用“按目录最后修改日期过滤”？").grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 4))
    tk.Checkbutton(root, text="启用", variable=enabled_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

    tk.Label(root, text="日期 (YYYY-MM-DD)：").grid(row=2, column=0, sticky="w", padx=10, pady=(8, 0))
    entry = tk.Entry(root, textvariable=date_var, width=18)
    entry.grid(row=2, column=1, sticky="w", padx=10, pady=(8, 0))

    result = {"ok": False}

    def on_ok():
        result["ok"] = True
        root.destroy()

    def on_cancel():
        result["ok"] = False
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=3, column=0, columnspan=2, pady=12)
    tk.Button(btn_frame, text="确定", width=10, command=on_ok).pack(side="left", padx=6)
    tk.Button(btn_frame, text="取消", width=10, command=on_cancel).pack(side="left", padx=6)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    entry.focus_set()
    root.mainloop()

    if not result["ok"]:
        return False, None

    enabled = bool(enabled_var.get())
    if not enabled:
        return False, None

    date_str = (date_var.get() or "").strip()
    if not date_str:
        return False, None

    return True, date_str

# 替换后的 main，使用脚本内常量且只检查一级子目录
def main():
    # 需要管理员权限运行
    print("--------------------------------请以管理员权限运行--------------------------------------")

    src = parse_args()
    
    dst = DST
    min_b = int(MIN_MB * 1024 * 1024)
    max_b = int(MAX_MB * 1024 * 1024)

    required_date = None

    _apply_env_overrides()
    
    if REQUIRED_MOD_TIME == 1:
        try:
            # 选框输入的日期格式为 YYYY-MM-DD
            enabled, date_str = prompt_for_mod_date_filter(REQUIRED_MOD_TIME, REQUIRED_MOD_DATE)
            if enabled:
                required_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            print(f"REQUIRED_MOD_DATE 格式错误: {REQUIRED_MOD_DATE}，应为 YYYY-MM-DD")
    elif REQUIRED_MOD_TIME == 2:
        enabled = True
        today = datetime.now().date()
        required_date = [today - timedelta(days=i) for i in range(7)]
    else:
        enabled = False
        date_str = None

        
    print(f"validate_paths result: {validate_paths(src, dst)}")
    if VERBOSE and validate_paths(src, dst):
        print(f"搜索源: {src}")
        print(f"目标目录: {dst}")
        print(f"大小范围: {MIN_MB} MB - {MAX_MB} MB")
        print(f"只检查一级子目录")
        print(f"干运行: {DRY_RUN}")
    else:
        print("退出")
        return
    
    candidates = find_timestamp_dirs(src, depth=1)
    # 按目录名解析的时间排序，最新的在最前面；解析失败的放到末尾
    candidates_sorted = sorted(
        candidates,
        key=lambda p: parse_timestamp_from_name(os.path.basename(p)) or datetime.min,
        reverse=True
    )

    if VERBOSE:
        print(f"找到符合命名的目录数量: {len(candidates_sorted)}（将按时间从新到旧检查）")

    matched_any = False
    # 从最新开始检查，找到第一个大小在范围内的就处理并退出
    for d in candidates_sorted:
        # 按目录“最后修改日期”过滤（不满足则直接跳过，不输出匹配结果）
        if required_date is not None:
            try:
                mdate = datetime.fromtimestamp(os.path.getmtime(d)).date()
            except Exception:
                if VERBOSE:
                    print(f"跳过: {d} 无法获取修改时间")
                continue

            
            if REQUIRED_MOD_TIME == 2:
                if mdate not in required_date:
                    if VERBOSE:
                        print(f"跳过: {d} 修改日期 {mdate} 不在最近7天内")
                    continue
            else:
                if mdate != required_date:
                    if VERBOSE:
                        print(f"跳过: {d} 修改日期 {mdate} != {required_date}")
                    continue
        try:
            size_b = get_folder_size_bytes(d)
        except Exception:
            size_b = 0
        if min_b <= size_b <= max_b:
            matched_any = True
            mb = bytes_to_mb(size_b)
            relname = os.path.basename(d)
            target = ensure_unique_dst(dst, relname)
            print(f"匹配: {d}  大小: {mb:.2f} MB  -> 复制到: {target}")

            copy_done = False
            final_basename = None

            if not DRY_RUN:
                try:
                    ok = copy_folder_explorer_like(d, target)
                    if not ok:
                        raise Exception("复制失败（robocopy/shutil 均失败）")
                    copy_done = True
                    # 复制成功后弹窗询问是否重命名；空输入或取消则不重命名
                    new_basename = prompt_for_rename(relname)
                    if new_basename:
                        new_target = ensure_unique_dst(dst, new_basename)
                        try:
                            os.rename(target, new_target)
                            print(f"已重命名: {target} -> {new_target}")
                            target = new_target
                            final_basename = new_basename
                        except Exception as e:
                            print(f"重命名失败: {target} -> {new_target} 错误: {e}")
                            final_basename = relname
                    else:
                        final_basename = relname
                except Exception as e:
                    print(f"复制失败: {d} -> {target} 错误: {e}")
            else:
                final_basename = relname

            # 如果实际完成了复制（非 DRY_RUN 且复制成功），在目标下创建 after/<同名>- 目录
            if not DRY_RUN and copy_done and final_basename:
                after_parent = os.path.join(dst, "after")
                after_csv_dir = os.path.join(after_parent, "csv")
                after_csv_dir = os.path.join(after_csv_dir, f"{final_basename}-")
                try:
                    os.makedirs(after_csv_dir, exist_ok=True)
                    if VERBOSE:
                        print(f"已创建 after 目录: {after_csv_dir}")
                except Exception as e:
                    print(f"创建 after 目录失败: {after_csv_dir} 错误: {e}")

            # 找到并处理第一个匹配后立即停止
            break
        else:
            if VERBOSE:
                print(f"跳过: {d} 大小 {bytes_to_mb(size_b):.2f} MB 不在范围内")

    if not matched_any:
        print("未找到满足条件的文件夹。")

if __name__ == "__main__":
    main()

