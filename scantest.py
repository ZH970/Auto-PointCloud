import cv2
import numpy as np
from PIL import Image, ImageGrab
from freegrab import grab_quadrilateral

def _order_points(pts: np.ndarray) -> np.ndarray:
    """4点排序：左上、右上、右下、左下"""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _load_image(image_path=None, mode=1):
    """
    mode=1: 本地图片
    mode=2: 剪贴板（支持：直接复制的图片 / 复制的文件路径列表 / 文本路径）
    """
    img = None

    if mode == 2:
        clip = ImageGrab.grabclipboard()
        if clip is None:
            print("剪贴板中无图像/路径")
            return None

        if isinstance(clip, Image.Image):
            img = cv2.cvtColor(np.array(clip.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(clip, list) and len(clip) > 0:
            img = cv2.imread(str(clip[0]))
        elif isinstance(clip, str) and clip.strip():
            img = cv2.imread(clip.strip())

    elif mode == 1 and image_path is not None:
        img = cv2.imread(image_path)
    elif mode == 3:
        # 截图（全屏）
        screenshot = ImageGrab.grab()
        img = cv2.cvtColor(np.array(screenshot.convert("RGB")), cv2.COLOR_RGB2BGR)

    if img is None:
        print("错误：无法读取图像（路径/剪贴板内容无效）")
        return None

    return img

def _red_mask_by_sample_rgb(hsv_img, sample_rgb=(250, 50, 80), h_tol=10, s_tol=90, v_tol=90):
    """
    用一个RGB样本色生成HSV阈值并得到mask（自动处理Hue环绕 0/179）
    方便用截图取色工具选取调试优化结果
    sample_rgb: (R,G,B)
    """
    # OpenCV 用 BGR，这里把 RGB -> BGR 再转 HSV
    bgr = np.uint8([[[sample_rgb[2], sample_rgb[1], sample_rgb[0]]]])
    h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0].tolist()

    s_low = max(0, int(s) - s_tol)
    v_low = max(0, int(v) - v_tol)
    s_high = min(255, int(s) + s_tol)
    v_high = min(255, int(v) + v_tol)

    h_low = int(h) - h_tol
    h_high = int(h) + h_tol

    masks = []
    if h_low < 0:
        # 例如 [-5, 5] => [0,5] U [175,179]
        masks.append(cv2.inRange(hsv_img, (0, s_low, v_low), (h_high, s_high, v_high)))
        masks.append(cv2.inRange(hsv_img, (180 + h_low, s_low, v_low), (179, s_high, v_high)))
    elif h_high > 179:
        # 例如 [175, 185] => [175,179] U [0,5]
        masks.append(cv2.inRange(hsv_img, (0, s_low, v_low), (h_high - 180, s_high, v_high)))
        masks.append(cv2.inRange(hsv_img, (h_low, s_low, v_low), (179, s_high, v_high)))
    else:
        masks.append(cv2.inRange(hsv_img, (h_low, s_low, v_low), (h_high, s_high, v_high)))

    red_mask = masks[0]
    for m in masks[1:]:
        red_mask = cv2.bitwise_or(red_mask, m)

    return red_mask, (int(h), int(s), int(v))

def _calculate_contour_center(cnt):
    """计算单个轮廓的中心（质心），返回 (cx, cy)，失败返回 None"""
    M = cv2.moments(cnt)
    if M["m00"] != 0:  # 避免除以0（无效轮廓）
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

def _find_isolated_contour(contours, distance_threshold=50):
    """
    从轮廓列表中筛选出孤立的轮廓（基于中心距离，平均距离最大的轮廓即为孤立轮廓）
    返回：孤立轮廓（单个cnt），若无法筛选返回面积最大的轮廓（兼容原逻辑）
    """
    if len(contours) == 1:
        # 只有一个轮廓，直接返回
        return contours[0]
    
    # 1. 计算所有轮廓的中心
    contour_centers = []
    valid_contours = []
    for cnt in contours:
        center = _calculate_contour_center(cnt)
        if center is not None:
            contour_centers.append(center)
            valid_contours.append(cnt)
    
    if len(valid_contours) < 1:
        return max(contours, key=cv2.contourArea)
    
    # 2. 计算每个轮廓与其他所有轮廓的平均距离
    avg_distances = []
    for i, (cx1, cy1) in enumerate(contour_centers):
        distances = []
        for j, (cx2, cy2) in enumerate(contour_centers):
            if i != j:  # 排除自身
                # 欧几里得距离
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                distances.append(dist)
        # 该轮廓的平均距离（越大表示越孤立）
        avg_dist = np.mean(distances) if distances else 0
        avg_distances.append(avg_dist)
    
    # 3. 筛选平均距离最大的轮廓（孤立轮廓）
    core_idx = np.argmax(avg_distances)
    core_contour = valid_contours[core_idx]
    core_center = contour_centers[core_idx]
    
    # 3. 筛选相近轮廓（距离核心轮廓 < distance_threshold 的所有轮廓）
    target_group = []
    for i, (cnt, center) in enumerate(zip(valid_contours, contour_centers)):
        cx, cy = center
        core_cx, core_cy = core_center
        # 计算当前轮廓与核心轮廓的距离
        dist_to_core = np.sqrt((cx - core_cx)**2 + (cy - core_cy)**2)
        if dist_to_core <= distance_threshold:
            target_group.append(cnt)
    
    # 4. 兜底：若相近轮廓为空，返回核心轮廓本身
    if not target_group:
        target_group = [core_contour]
    
    return target_group

def detect_target(image_path=None, mode=1, visualize=True):
    """
    提取标靶红色边缘轮廓，并用“方形”拟合，返回角点坐标（原图像素坐标）。
    优化：基于标靶孤立性筛选轮廓，避免同色干扰影响拟合结果。

    返回：
    - result_img: 画了拟合方形的图
    - box_points: 4个点坐标，顺序(左上,右上,右下,左下)，shape=(4,2)，dtype=int
    """
    img = _load_image(image_path=image_path, mode=mode)
    if img is None:
        return None, None

    result_img = img.copy()

    # 1) HSV 提取红色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_mask, hsv_center = _red_mask_by_sample_rgb(
        hsv,
        sample_rgb=(250, 50, 80),
        h_tol=13,    # 色相容差：越大越宽松
        s_tol=130,   # 饱和度容差
        v_tol=100    # 亮度容差
    )

    # 2) 形态学去噪/补边
    kernel = np.ones((3, 3), np.uint8)
    red_mask_optimized = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=5)

    # 3) 找轮廓 + 筛选有效轮廓
    contours, _ = cv2.findContours(red_mask_optimized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未检测到红色边框轮廓")
        if visualize:
            cv2.imshow("Original", result_img)
            cv2.imshow("Detected_Red", red_mask_optimized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return result_img, None

    # 过滤太小的噪声
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 300]
    if not valid_contours:
        print("红色轮廓存在，但都太小（被过滤）")
        if visualize:
            cv2.imshow("Original", result_img)
            cv2.imshow("Detected_Red", red_mask_optimized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return result_img, None

    # 4) 筛选孤立轮廓（标靶），替代原有的“所有轮廓合集”
    target_contour = _find_isolated_contour(valid_contours, distance_threshold=100)
    # 若需可视化孤立轮廓与其他轮廓的区别，可注释下方代码
    # 绘制所有有效轮廓（蓝）+ 孤立标靶轮廓（红）
    cv2.drawContours(result_img, valid_contours, -1, (255, 0, 0), 2)  # 所有有效轮廓（蓝）
    cv2.drawContours(result_img, target_contour, -1, (0, 0, 255), 3)  # 孤立标靶轮廓（红，粗线）

    # 5) 方形拟合：仅基于孤立标靶轮廓（替换原有的 all_pts 合集）
    all_target_pts = np.vstack(target_contour).astype(np.float32)
    rect = cv2.minAreaRect(all_target_pts)  # ((cx,cy),(w,h),angle)
    (cx, cy), (w, h), angle = rect

    square_rect = ((cx, cy), (float(w), float(h)), angle)
    box = cv2.boxPoints(square_rect)  # 4x2 float
    box = _order_points(box)
    box_int = np.round(box).astype(int)

    # 6) 可视化：画拟合方形+写坐标
    cv2.polylines(result_img, [box_int], isClosed=True, color=(0, 255, 0), thickness=2)  # 拟合方形(绿)

    for i, (x, y) in enumerate(box_int):
        cv2.circle(result_img, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv2.putText(result_img, f"P{i}:{x},{y}", (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    print("成功检测到孤立标靶红色边缘，并完成方形拟合。")
    print("方形角点坐标(左上,右上,右下,左下)：")
    print(box_int.tolist())

    if visualize:
        cv2.imshow("Original+FitSquare", result_img)
        cv2.imshow("Detected_Red", red_mask_optimized)
        cv2.waitKey(2500)
        cv2.destroyAllWindows()

    return result_img, box_int

def warp_by_bbox(img_bgr: np.ndarray, bbox: np.ndarray, output_size=(400, 400)) -> np.ndarray:
    pts = np.asarray(bbox, dtype=np.float32)
    pts = _order_points(pts)

    w, h = output_size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img_bgr, M, (w, h))
    return warped

def match_target_by_template(
    template_path: str,
    image_path=None,
    mode=1,
    visualize=True,
    nfeatures=5000,
    ratio=0.8,
    min_matches=12,
    ransac_thresh=5.0,
):
    """
    方案D：模板特征匹配（ORB） + RANSAC Homography
    返回：
    - result_img: 在目标图上画出投影四边形
    - box_int: 目标图中的四角点(左上,右上,右下,左下)，shape=(4,2)，int
    - warped: 将目标图透视矫正到“模板坐标系”的结果（大小=模板大小）
    - H: 单应矩阵（template -> image）
    """
    template = cv2.imread(template_path)
    if template is None:
        print(f"错误：无法读取模板图片: {template_path}")
        return None, None, None, None
    print(isinstance(template, np.ndarray))

    # image_path 既可以是路径(str)，也可以直接传入图像(np.ndarray)
    if isinstance(image_path, np.ndarray):
        img = image_path.copy()
    elif isinstance(image_path, Image.Image):
        img = cv2.cvtColor(np.array(image_path.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        img = _load_image(image_path=image_path, mode=mode)
    
    if img is None:
        return None, None, None, None

    gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_i = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(
        nfeatures=5000,      # 增加特征点数量
        scaleFactor=1.1,     # 更细的尺度层次
        nlevels=12,          # 更多尺度层
        edgeThreshold=15,    # 降低边缘阈值
        patchSize=31,        # 增大patch大小
    )
    kp1, des1 = orb.detectAndCompute(gray_t, None)
    kp2, des2 = orb.detectAndCompute(gray_i, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("错误：特征点不足（模板或目标图太模糊/太纯色/分辨率不合适）")
        return img, None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < float(ratio) * n.distance:
            good.append(m)

    if len(good) < int(min_matches):
        print(f"匹配点不足：good={len(good)} < min_matches={min_matches}")
        # 在 match_target_by_template 函数中添加
        print(f"模板特征点数量: {len(kp1)}")
        print(f"目标图特征点数量: {len(kp2)}")
        print(f"初始匹配对数量: {len(knn)}")
        print(f"通过比值测试的匹配: {len(good)}")
        return img, None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, float(ransac_thresh))
    if H is None:
        print("错误：Homography 计算失败（内点不足或匹配退化）")
        return img, None, None, None

    ht, wt = gray_t.shape[:2]
    corners = np.float32([[0, 0], [wt - 1, 0], [wt - 1, ht - 1], [0, ht - 1]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(4, 2)

    proj = _order_points(proj)
    box_int = np.round(proj).astype(int)

    result_img = img.copy()
    cv2.polylines(result_img, [box_int], isClosed=True, color=(0, 255, 0), thickness=2)
    for i, (x, y) in enumerate(box_int):
        cv2.circle(result_img, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv2.putText(result_img, f"P{i}:{x},{y}", (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    H_inv = np.linalg.inv(H)
    warped = cv2.warpPerspective(img, H_inv, (wt, ht))

    inlier_list = None
    if inlier_mask is not None:
        inlier_list = inlier_mask.ravel().tolist()

    if visualize:
        match_vis = cv2.drawMatches(
            template, kp1, img, kp2, good, None,
            matchesMask=inlier_list,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imshow("Template", template)
        cv2.imshow("Matches(Inliers)", match_vis)
        cv2.imshow("DetectedQuad", result_img)
        cv2.imshow("WarpedToTemplate", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("模板匹配成功，角点(左上,右上,右下,左下)：")
    print(box_int.tolist())
    return result_img, box_int, warped, H

# 示例：
# 1) 本地图片
# detect_target("target_image.jpg", mode=1, visualize=True)
# 2) 剪贴板图片（先复制图片/截图/图片文件路径）
if __name__ == "__main__":
    img, bbox = detect_target(mode=2, visualize=True)
    img = _load_image(mode=2)              # 读剪贴板原图
    vis, bbox = detect_target(mode=2, visualize=True)  # 你现在 detect_target 会再读一次图；后面我再教你怎么改成传 img
    warped = warp_by_bbox(img, bbox, (400, 400))

    cv2.imshow("Extracted_Quadrilateral", warped)
    match_target_by_template("image.png", warped, mode=1, visualize=True)