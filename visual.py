import argparse
import time

import cv2
import numpy as np
from typing import Optional
from PIL import ImageGrab


def _build_detector_params():
    aruco = cv2.aruco
    try:
        params = aruco.DetectorParameters_create()
    except AttributeError:
        params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 2
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.cornerRefinementMethod = getattr(aruco, "CORNER_REFINE_SUBPIX", 1)
    params.cornerRefinementWinSize = 5
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 1
    return params


def get_dictionary(name: str = "DICT_4X4_1000"):
    aruco = cv2.aruco
    key = name.upper()
    if key == "AUTO":
        key = "DICT_4X4_1000"
    if not hasattr(aruco, key):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(getattr(aruco, key))


def build_board(squares_x: int, squares_y: int, square_length: float, marker_length: float, dictionary):
    aruco = cv2.aruco
    try:
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)
    except AttributeError:
        board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board


def detect_charuco(image, board, dictionary):
    aruco = cv2.aruco
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)
    params = _build_detector_params()
    corners, ids, _ = aruco.detectMarkers(gray_enh, dictionary, parameters=params)
    if ids is None or len(ids) == 0:
        return None

    detection = {
        "marker_corners": corners,
        "marker_ids": ids,
        "charuco_corners": None,
        "charuco_ids": None,
        "charuco_retval": 0
    }

    try:
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )
    except cv2.error:
        retval, charuco_corners, charuco_ids = 0, None, None

    if (
        retval is not None
        and retval >= 4
        and charuco_corners is not None
        and charuco_ids is not None
    ):
        detection["charuco_corners"] = charuco_corners
        detection["charuco_ids"] = charuco_ids
        detection["charuco_retval"] = int(retval)

    return detection

def detect_two_diagonal_filled_squares(
    bgr: np.ndarray,
    min_area_ratio: float = 0.005,   # 占整图面积比例，过滤小噪声
    max_area_ratio: float = 0.8,     # 过滤几乎全屏的大块
    aspect_tol: float = 0.18,        # 宽高比允许偏差（0.18≈ 1:1 ±18%）
    fill_thresh: float = 0.90,       # “实心程度”：黑像素填充率阈值
    diag_balance_tol: float = 0.6,   # |dx| 和 |dy| 的平衡程度（越小越严格）
    morph_ksize: int = 5,            # 闭运算核，填补黑块里的小亮斑
    debug_path: Optional[str] = None,
):
    """
    识别两个对角线排列的黑色实心正方形。
    返回: (center1, center2, outer_square_bbox) 或 None
      - center1/2: (cx, cy) float
      - outer_square_bbox: (x0, y0, x1, y1) int，最小外接“正方形”(轴对齐)
    """
    if bgr is None or bgr.size == 0:
        return None

    h, w = bgr.shape[:2]
    img_area = float(h * w)
    min_area = img_area * float(min_area_ratio)
    max_area = img_area * float(max_area_ratio)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 黑色前景：用 OTSU 自动阈值更稳（黑->1，白->0）
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if morph_ksize and morph_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _hier = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)
        if ww <= 1 or hh <= 1:
            continue

        aspect = ww / float(hh)
        if abs(aspect - 1.0) > aspect_tol:
            continue

        # 用“填充率”判断是否实心：轮廓mask中黑前景像素占 bounding box 的比例
        mask = np.zeros((hh, ww), dtype=np.uint8)
        cnt_shift = cnt.copy()
        cnt_shift[:, 0, 0] -= x
        cnt_shift[:, 0, 1] -= y
        cv2.drawContours(mask, [cnt_shift], -1, 255, thickness=-1)

        roi_bw = bw[y:y+hh, x:x+ww]
        filled = cv2.countNonZero(cv2.bitwise_and(roi_bw, mask))
        fill_ratio = filled / float(ww * hh)

        if fill_ratio < fill_thresh:
            continue

        # 中心点（用 boundingRect 中心即可，够稳）
        cx = x + ww / 2.0
        cy = y + hh / 2.0

        candidates.append({
            "area": area,
            "rect": (x, y, ww, hh),
            "center": (cx, cy),
            "fill": fill_ratio,
        })

    if len(candidates) < 2:
        return None

    # 取面积最大的两个
    candidates.sort(key=lambda d: d["area"], reverse=True)
    a, b = candidates[0], candidates[1]
    (cax, cay), (cbx, cby) = a["center"], b["center"]

    dx = abs(cax - cbx)
    dy = abs(cay - cby)
    if dx < 1 or dy < 1:
        return None

    # 对角线排列：x、y都应有明显差异，且 dx/dy 不要极端失衡
    balance = abs(dx - dy) / max(dx, dy)  # 0=很像正对角，1=很偏
    if balance > diag_balance_tol:
        return None

    # 固定顺序：返回 (左上, 右下)
    if (cax + cay) <= (cbx + cby):
        top_left = a
        bottom_right = b
    else:
        top_left = b
        bottom_right = a

    # 两个正方形的最小外接“正方形”bbox
    x1, y1, w1, h1 = top_left["rect"]
    x2, y2, w2, h2 = bottom_right["rect"]
    minx = min(x1, x2)
    miny = min(y1, y2)
    maxx = max(x1 + w1, x2 + w2)
    maxy = max(y1 + h1, y2 + h2)

    bwid = maxx - minx
    bhei = maxy - miny
    side = int(np.ceil(max(bwid, bhei)))

    cx_all = (minx + maxx) / 2.0
    cy_all = (miny + maxy) / 2.0
    x0 = int(np.floor(cx_all - side / 2))
    y0 = int(np.floor(cy_all - side / 2))
    x1o = x0 + side
    y1o = y0 + side

    # clip 到图像范围
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1o = min(w, x1o)
    y1o = min(h, y1o)

    if debug_path:
        dbg = bgr.copy()
        for item, color in [(top_left, (0, 255, 0)), (bottom_right, (0, 255, 255))]:
            x, y, ww, hh = item["rect"]
            cv2.rectangle(dbg, (x, y), (x+ww, y+hh), color, 2)
            cx, cy = item["center"]
            cv2.drawMarker(dbg, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 18, 2)
        cv2.rectangle(dbg, (x0, y0), (x1o, y1o), (255, 0, 0), 2)
        cv2.imwrite(debug_path, dbg)

    return top_left["center"], bottom_right["center"], (x0, y0, x1o, y1o)

def unsharp_mask(image, k=1.5):
    """对 OpenCV 图像做反遮罩锐化，并尽量保持输入“原格式”。

    - 保持通道数不变（BGR/灰度都支持）
    - 保持 dtype 不变：
      - uint8/uint16 等整数：会 clip 到类型范围再 cast 回原 dtype
      - float32/float64：不强行 clip（除非你自己保证范围），返回同 dtype
    """
    if image is None:
        return image

    src = image
    src_dtype = src.dtype

    # 统一用 float32 做计算，避免 uint8 下溢/溢出
    src_f = src.astype(np.float32, copy=False)
    blurred = cv2.GaussianBlur(src_f, (9, 9), 10.0)
    detail = src_f - blurred
    sharpened = src_f + float(k) * detail

    # 恢复到原 dtype
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        sharpened = np.clip(sharpened, info.min, info.max)
        return sharpened.astype(src_dtype)
    # float：保持 dtype；不做 clip，避免误伤 HDR/归一化数据
    return sharpened.astype(src_dtype, copy=False)


def sharpen_keep_format(image: np.ndarray, k: float = 1.5) -> np.ndarray:
    """更直观的别名：锐化并保持原 dtype/通道格式。"""
    return unsharp_mask(image, k=k)

def simulate_pointcloud_sampling(
    bgr: np.ndarray,
    downscale: float = 0.6,     # 0~1，越小越“块状”
    levels: int = 6,             # 灰度离散级数，越小越离散
    dropout: float = 0.1,       # 0~1，丢点比例，越大越稀疏
    blur_ksize: int = 3,         # 0 或奇数，比如 3/5
    noise_sigma: float = 6.0,    # 噪声强度(像素)
    jpeg_quality: int | None = 60,  # 1~100，越小压缩越狠；None=不做jpeg
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    img = bgr.copy()
    cv2.imshow("img", img)
    time.sleep(2)
    h, w = img.shape[:2]

    # 1) 降采样 + 最近邻放大（块状/锯齿）
    if downscale is not None and 0 < downscale < 1:
        sh = max(1, int(h * downscale))
        sw = max(1, int(w * downscale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2) 灰度 & 均衡（点云窗口常见）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3) 离散化（levels 档）
    if levels is not None and levels >= 2:
        step = 255 // (levels - 1)
        gray = (np.round(gray / step) * step).clip(0, 255).astype(np.uint8)

    # 4) 稀疏化：随机丢像素（更像点云“缺点”）
    if dropout is not None and 0 < dropout < 1:
        mask_keep = rng.random(gray.shape) > dropout
        gray = (gray * mask_keep.astype(np.uint8)).astype(np.uint8)

    # 5) 模糊 + 噪声（轻微即可）
    if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if noise_sigma and noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, size=gray.shape).astype(np.float32)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6) JPEG 压缩伪影（可选）
    if jpeg_quality is not None:
        ok, enc = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if ok:
            gray = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    return gray

def compute_center(points: np.ndarray):
    if points.size == 0:
        return (0.0, 0.0)
    return tuple(points.mean(axis=0).tolist())

def detect_aruco_markers(image, dictionary, simulate_pointcloud: bool = False):
    aruco = cv2.aruco
    if simulate_pointcloud:
        gray_enh = simulate_pointcloud_sampling(image,jpeg_quality=None)
        cv2.imwrite("simulated_pointcloud.png", gray_enh)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enh = clahe.apply(gray)
        cv2.imwrite("gray_enh.png", gray_enh)
    params = _build_detector_params()
    corners, ids, _ = aruco.detectMarkers(gray_enh, dictionary, parameters=params)
    if ids is None or len(ids) == 0:
        return None, None
    return corners, ids.flatten()

def annotate(image, detection, centers, bbox=None):
    """
    intro:在图像上绘制检测到的 ArUco 标记和中心点
    说明:在给定的图像上绘制检测到的 ArUco 标记、它们的角点、中心点以及可选的边界框。
    
    :param image: 输入图像，通常是彩色图像（BGR格式）
    :param detection: 包含检测到的标记信息的字典，包含标记角点和ID
    :param centers: 标记中心点的列表或单个中心点的元组
    :param bbox: 可选的边界框列表，格式为 [(x0, y0, x1, y1), ...]
    :return: 带有注释的图像
    :type image: np.ndarray
    :rtype: np.ndarray
    """

    aruco = cv2.aruco
    annotated = image.copy()

    # centers 既支持 (x,y) 也支持 [(x,y), (x,y), ...]
    if centers is None:
        centers = []
    elif isinstance(centers, tuple) and len(centers) == 2:
        centers = [centers]

    marker_ids = detection["marker_ids"].flatten()
    aruco.drawDetectedMarkers(annotated, detection["marker_corners"], marker_ids)

    for marker_id, marker_corner in zip(marker_ids, detection["marker_corners"]):
        pts = marker_corner.reshape(-1, 2)
        for idx, (x, y) in enumerate(pts):
            px, py = int(round(float(x))), int(round(float(y)))
            cv2.circle(annotated, (px, py), 3, (255, 0, 0), -1)
            cv2.putText(
                annotated,
                f"{int(marker_id)}-{idx}",
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA
            )

    for c_idx, (cx, cy) in enumerate(centers):
        ix, iy = int(round(float(cx))), int(round(float(cy)))
        cv2.drawMarker(
            annotated, (ix, iy), (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2
        )
        cv2.putText(
            annotated,
            f"C{c_idx}",
            (ix + 6, iy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        if bbox is not None:
            for (x0, y0, x1, y1) in bbox:
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)

    return annotated

def _homography_reproj_error(H, obj_pts, img_pts):
    """计算重投影误差（均值像素）"""
    proj = cv2.perspectiveTransform(obj_pts.reshape(-1,1,2).astype(np.float32), H).reshape(-1,2)
    err = np.linalg.norm(proj - img_pts.reshape(-1,2), axis=1)
    return float(np.mean(err))

def solve_board_center_bbox_from_two_markers(image, single_record, ids_pair, marker_length, separation,
                                             ransac_thresh=3.0, padding=0):
    """
    intro:识别corners dict的两个id组成的板子的外接框
    single_record, [dict[int]:list[4]=[tuple(x,y)*4]], 一块板子上的id与角点坐标映射
    ids_pair: (idA, idB) 两个标记的ID
    marker_length: L，单个marker边长（同单位）
    separation: sep，标记之间的缝隙宽度（同单位）
    返回: center(x,y), bbox(x0,y0,x1,y1) 或 (None, None)
    """
    corners_dict = {}
    for mid in ids_pair:
        pts = single_record.get(mid, [])
        if pts:
            corners_dict[mid] = np.array(pts, dtype=np.float32).reshape(1, 4, 2)

    L = float(marker_length); sep = float(separation)
    idA, idB = int(ids_pair[0]), int(ids_pair[1])
    if idA not in corners_dict or idB not in corners_dict:
        return None, None

    img_corners_A = corners_dict[idA].reshape(-1,2).astype(np.float32)  # tl,tr,br,bl
    img_corners_B = corners_dict[idB].reshape(-1,2).astype(np.float32)

    # 2×2 棋盘四种候选布局（A 和 B 分别落在对角的两个格子）
    # 棋盘平面坐标系：原点在左上角，X向右，Y向下
    # 每个格子的大小是 L，两个格子之间的缝隙是 sep，marker比棋盘边缘内缩 sep/2
    W = 2*L + 2*sep
    Hh = W

    layouts = [
        # A 在左上(0,0)，B 在右下(L+sep, L+sep)
        ((sep/2,sep/2), (L+3*sep/2, L+3*sep/2)),
        # A 在右上(L+sep,0)，B 在左下(0, L+sep)
        ((L+3*sep/2,sep/2), (sep/2, L+3*sep/2)),
        # A 在左下(0,L+sep)，B 在右上(L+sep,0)
        ((sep/2, L+3*sep/2), (L+3*sep/2,sep/2)),
        # A 在右下(L+sep,L+sep)，B 在左上(0,0)
        ((L+3*sep/2, L+3*sep/2), (sep/2,sep/2)),
    ]

    # RANSAC 估计单应矩阵，分析当前布局
    best = None
    for (Ax, Ay), (Bx, By) in layouts:
        # OpenCV角点顺序: tl, tr, br, bl
        objA = np.array([[Ax,Ay],[Ax+L,Ay],[Ax+L,Ay+L],[Ax,Ay+L]], np.float32)
        objB = np.array([[Bx,By],[Bx+L,By],[Bx+L,By+L],[Bx,By+L]], np.float32)
        obj_pts = np.vstack([objA, objB])        # 8×2
        img_pts = np.vstack([img_corners_A, img_corners_B])  # 8×2

        Hmat, mask = cv2.findHomography(obj_pts, img_pts, cv2.RANSAC, float(ransac_thresh))
        if Hmat is None:
            continue
        err = _homography_reproj_error(Hmat, obj_pts, img_pts)
        if (best is None) or (err < best[0]):
            best = (err, Hmat)

    if best is None:
        return None, None

    Hmat = best[1]
    # 计算外接矩形四角与中心
    obj_rect = np.array([[0,0],[W,0],[W,Hh],[0,Hh]], np.float32).reshape(-1,1,2)
    img_rect = cv2.perspectiveTransform(obj_rect, Hmat).reshape(-1,2)
    obj_center = np.array([[W/2, Hh/2]], np.float32).reshape(-1,1,2)
    img_center = cv2.perspectiveTransform(obj_center, Hmat).reshape(-1,2)[0]
    cx, cy = float(img_center[0]), float(img_center[1])

    # bbox（右下角开区间）
    x0 = max(0, int(np.floor(np.min(img_rect[:,0])) - padding))
    y0 = max(0, int(np.floor(np.min(img_rect[:,1])) - padding))
    x1 = int(np.ceil(np.max(img_rect[:,0])) + padding)
    y1 = int(np.ceil(np.max(img_rect[:,1])) + padding)
    return (cx, cy), (x0, y0, x1, y1)

def grab_screen():
    screenshot = ImageGrab.grab()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def get_board_centers_from_screen(
    groups=None,
    dictionary_name="DICT_4X4_1000",
): 
    """
    intro:从屏幕截图中检测并计算指定组的 ArUco 标记中心点
    说明:捕获当前屏幕内容并检测指定组的 ArUco 标记，计算它们的中心点坐标。
    返回: centers（组名到中心点坐标的映射），records（标记ID到角点坐标的映射），image（截图图像），detection（检测结果字典）
    groups: 字典，键为组名，值为该组包含的标记ID列表
    dictionary_name: ArUco 字典名称，用于标记检测
    """
    image = grab_screen()
    # 可按需启用：对截图做锐化，但保持 BGR + dtype 不变
    image = sharpen_keep_format(image, k=1.2)
    centers, records, detection = get_board_centers_from_image(
        image=image,
        groups=groups,
        dictionary_name=dictionary_name,
    )
    return centers, records, image, detection

def get_board_centers_from_image(
    image,
    groups=None,
    dictionary_name="DICT_4X4_1000",
):
    """
    intro:从图像中检测并计算指定组的 ArUco 标记中心点
    image: 输入图像，通常是彩色图像（BGR格式）
    groups: 字典，键为组名，值为该组包含的标记ID列表
    dictionary_name: ArUco 字典名称，用于标记检测
    返回: centers（组名到中心点坐标的映射），records（标记ID到角点坐标的映射），detection（检测结果字典）
    """
    if groups is None:
        groups = {
            "boardA_392_393": [392, 393],
            "boardB_394_395": [394, 395],
        }

    dictionary = get_dictionary(dictionary_name)
    #corners, ids = detect_aruco_markers(image,dictionary)
    corners, ids = detect_aruco_markers(image, dictionary, True)
    if ids is None:
        empty_detection = {
            "marker_corners": [],
            "marker_ids": np.empty((0, 1), dtype=np.int32),
        }
        return {}, {}, empty_detection

    # 组装成 annotate 能用的 detection 格式
    detection = {
        "marker_corners": corners,
        "marker_ids": ids.reshape(-1, 1).astype(np.int32),
    }

    target_ids = {int(mid) for mids in groups.values() for mid in mids}
    records = {mid: [] for mid in target_ids}

    for marker_id, marker_corner in zip(ids, corners):
        mid = int(marker_id)
        if mid not in records:
            continue
        pts = marker_corner.reshape(-1, 2)  # (4,2)
        records[mid] = [(float(x), float(y)) for (x, y) in pts]

    centers = {}
    for name, mids in groups.items():
        mids = [int(m) for m in mids]
        if any(len(records.get(mid, [])) == 0 for mid in mids):
            continue

        all_pts = []
        for mid in mids:
            all_pts.extend(records[mid])

        centers[name] = compute_center(np.array(all_pts, dtype=np.float32))

    return centers, records, detection
    
def main():
    parser = argparse.ArgumentParser(description="Detect ArUco board centers from image/screen.")
    parser.add_argument("--image", default=None, help="Image path; omit to capture the current screen.")
    parser.add_argument("--dictionary", default="DICT_4X4_1000", help="ArUco dictionary name or AUTO.")
    parser.add_argument("--show", type=bool, default=True, help="Display annotated detection window.")
    args = parser.parse_args()

    groups = {
        "boardA_392_393": [392, 393],
        "boardB_394_395": [394, 395],
    }
    target_ids = [392,393,394,395]
    L = 8
    sep = 3

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        centers, records, detection = get_board_centers_from_image(
            image=image,
            groups=groups,
            dictionary_name=args.dictionary,
        )
    else:
        centers, records, image, detection = get_board_centers_from_screen(
            groups=groups,
            dictionary_name=args.dictionary,
        )

    newcenter1, bbox1 = solve_board_center_bbox_from_two_markers(image, records, (392, 393), L, sep)
    newcenter2, bbox2 = solve_board_center_bbox_from_two_markers(image, records, (394, 395), L, sep)

    if not centers:
        print("No target boards detected.")
    else:
        for idx, (name, (cx, cy)) in enumerate(centers.items()):
            print(f"{idx} {name} center x={cx:.3f} y={cy:.3f}")
    if newcenter1:
        print(newcenter1, bbox1)
    if newcenter2:
        print(newcenter2, bbox2)
    if args.show:
        # 只画中心点：传 centers.values() 即可（不会有 None）
        center_list = list(centers.values())
        bboxes = [b for b in (bbox1, bbox2) if b is not None]
        annotated = annotate(image, detection, center_list, bboxes if bboxes else None)
        res = detect_two_diagonal_filled_squares(image, debug_path="diagonal_squares_debug.png")
        if res:
            c1, c2, bbox = res
            print(f"Detected diagonal filled squares centers: {c1}, {c2}, bbox: {bbox}")
        cv2.imshow("ArUco board centers", annotated)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if not hasattr(cv2, "aruco"):
        raise ImportError("OpenCV build missing aruco module; install opencv-contrib-python.")
    main()