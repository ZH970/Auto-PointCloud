import argparse
import cv2
import numpy as np
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


def compute_center(points: np.ndarray):
    if points.size == 0:
        return (0.0, 0.0)
    return tuple(points.mean(axis=0).tolist())

def detect_aruco_markers(image, dictionary):
    aruco = cv2.aruco
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)
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
    corners, ids = detect_aruco_markers(image, dictionary)
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
        cv2.imshow("ArUco board centers", annotated)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if not hasattr(cv2, "aruco"):
        raise ImportError("OpenCV build missing aruco module; install opencv-contrib-python.")
    main()