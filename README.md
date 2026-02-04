## 使用教程
**注意由于LidarME的权限，程序必须以管理员身份运行，且注意插上密钥再启动程序**
### main
**点云文件夹自动导入**

REQUIRED_MOD_TIME 工作模式 1表示允许按修改日期过滤，且手动输入日期，2是只匹配最近7天,0表示不启用

有需要可以查看main的配置区，可以修改点云文件大小范围，调试信息打印等

### auto
**点云处理自动化**

自动从POINTCLOUD_ROOT识别新生成的文件夹导入

    # 一次只能输入10个文件夹，防止软件无法判断是结算没完成还是在下一页
    # pending = list(list_pending_folders(POINTCLOUD_ROOT, PROCESSED_MARK_DIR))
有多新：默认是输入今天内生成的文件夹

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
            if modified_date != today: # 这里判断今日内
                continue
            if folder.name in processed:
                continue
            yield folder

想要手动指定文件夹输入：
注释掉LINE677：`   pending = list(list_pending_folders(POINTCLOUD_ROOT, PROCESSED_MARK_DIR))`
然后在后面加：

    pending = []
    # for test only
    pending.append(Path(ROOT_STR + "\\" + "0973"))
    pending.append(Path(ROOT_STR + "\\" + "1005"))


然后要跳过导入文件夹部分：
注释掉LINE603-617这部分：

    for folder in folders_list:
        logging.info("开始导入: %s", folder)
        click_button(window, NEW_TASK_BUTTON_TEXT, "Text", 2)
        #time.sleep(WAIT_ACTION)
        click_button(window, "选择文件夹", "Button", 2)
        dialog = wait_dialog(app, SELECT_DIALOG_TITLE_RE, WAIT_DIALOG)
        choose_folder(dialog, folder.name, folder.parent.as_posix())
        click_button(window,title="RTK校准")
        click_button(window,title="确认")
        time.sleep(3.0)
        window = ensure_main_window(app)
        logging.info("导入完成: %s", folder)
        time.sleep(0.2)

    time.sleep(1.0)

## 项目功能
- 这个工作区主要是围绕 “Lidar ME 点云采集 → 拷贝/整理 → 在 Lidar ME 软件里批处理导入与导出 → 后处理（Excel/图片）” 的一套自动化脚本集合。

- main.py（**SD 卡导入器**）
  - 通过卷标 `SRC_LABEL` 找到插入的设备盘（SD 卡/读卡器），在源盘根目录（或子路径）里找符合时间戳命名的文件夹（如 `2025-11-11_15-50-12`）。
  - 按文件夹大小范围、修改日期（可选最近 7 天/指定日期）过滤，找到匹配项后复制到 `DST`。
  - 复制优先用 `robocopy`，失败回退 `shutil`；复制后可弹窗重命名；并在目标下创建 `after/csv/<folder>-` 这种后续处理目录结构。

- auto.py（**Lidar ME UI 自动化流水线**）
  - 用 `pywinauto` 操作 Lidar ME 软件：创建任务/选择文件夹/确认/翻页定位任务行/进入任务。
  - `click_view_for_task()` 负责在任务列表里找到对应任务行（支持 `name(1)` 这种），并优先选择带“今天 HH:MM”时间控件的那行。
  - `solve()` 负责在软件里跑“坐标转换/处理/导出坐标”等流程；中间会做截图/标靶检测（调用 scantest.py 的 `detect_target`、模板匹配、透视矫正等），保存到 `after/img`；然后提示你可以开始手动选点（弹窗 1s 自动关闭），监听你连续两次回车后继续导出。

- scantest.py（**标靶检测与匹配**）
  - 从截图/剪贴板/全屏抓图中提取红色标靶边框轮廓，拟合四边形角点；也提供模板特征匹配（ORB+RANSAC）与 `warp_by_bbox` 透视矫正。

- freegrab.py（**四边形截图工具**）
  - 给四个屏幕坐标点，先矩形截图再透视变换，得到矫正后的标靶局部图（PIL）。

- keylistener.py（**输入同步**）
  - 全局监听键盘：检测 1 秒内连续两次回车，作为“我已选点完成”的信号，驱动 auto.py 继续后续自动步骤。

- points2SLAM_Output.py：把 `points.csv` 等结果写入 Excel（包含对图片/关系的处理）。

- visual.py：偏视觉算法/调试工具（ArUco/模拟点云效果等），更多是辅助模块。

## 工作流
`SD卡(原始点云) -> main.py复制到DST -> auto.py导入/处理/导出 -> after/csv + after/img -> points2SLAM_Output.py写Excel`。

## 工作目录
点云工作目录

    {ROOT_STR}
        |-XXXX #点云文件夹
        |-after
            |-csv
                |-XXXX- #坐标转换导出文件夹
            |-img
                |-XXXX.png #截图存放

## 未来展望
试用阶段，请试用者留意要改进之处并提出，标靶搜寻算法如果有想法可以说一下
之后会提供前端ui整合（可能）