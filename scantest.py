import cv2
import numpy as np
from PIL import Image, ImageGrab

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

def _find_isolated_contour(contours):
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
    isolated_idx = np.argmax(avg_distances)
    return valid_contours[isolated_idx]

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
    red_mask_optimized = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

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
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 200]
    if not valid_contours:
        print("红色轮廓存在，但都太小（被过滤）")
        if visualize:
            cv2.imshow("Original", result_img)
            cv2.imshow("Detected_Red", red_mask_optimized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return result_img, None

    # 4) 筛选孤立轮廓（标靶），替代原有的“所有轮廓合集”
    target_contour = _find_isolated_contour(valid_contours)
    # 若需可视化孤立轮廓与其他轮廓的区别，可注释下方代码
    # 绘制所有有效轮廓（蓝）+ 孤立标靶轮廓（红）
    cv2.drawContours(result_img, valid_contours, -1, (255, 0, 0), 2)  # 所有有效轮廓（蓝）
    cv2.drawContours(result_img, [target_contour], -1, (0, 0, 255), 3)  # 孤立标靶轮廓（红，粗线）

    # 5) 方形拟合：仅基于孤立标靶轮廓（替换原有的 all_pts 合集）
    rect = cv2.minAreaRect(target_contour)  # ((cx,cy),(w,h),angle)
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


# 示例：
# 1) 本地图片
# detect_target("target_image.jpg", mode=1, visualize=True)
# 2) 剪贴板图片（先复制图片/截图/图片文件路径）
if __name__ == "__main__":
    detect_target(mode=2, visualize=True)