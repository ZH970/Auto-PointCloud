from PIL import ImageGrab, Image
import cv2
import numpy as np

def _order_points(pts: np.ndarray) -> np.ndarray:
    """4点排序：左上、右上、右下、左下（复用你之前的函数，保证坐标一致性）"""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def grab_quadrilateral(quad_points, output_size=(400, 400), visualize=True):
    """
    自定义四边形截图（透视变换矫正）
    :param quad_points: 四边形4个顶点坐标，列表格式 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]（屏幕坐标，左上为原点）
    :param output_size: 矫正后的图像尺寸 (宽, 高)
    :param visualize: 是否可视化结果
    :return: 矫正后的四边形截图（PIL.Image 格式）
    """
    # 步骤1：验证输入坐标（必须是4个点）
    if len(quad_points) != 4:
        print("错误：必须传入4个顶点坐标")
        return None
    quad_pts = np.array(quad_points, dtype=np.float32)
    if quad_pts.shape != (4, 2):
        print("错误：坐标格式必须为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]")
        return None

    # 步骤2：找到四边形的最小矩形包围盒（获取截图的bbox）
    x_coords = quad_pts[:, 0]
    y_coords = quad_pts[:, 1]
    x1 = int(np.min(x_coords))
    y1 = int(np.min(y_coords))
    x2 = int(np.max(x_coords))
    y2 = int(np.max(y_coords))

    # 步骤3：截取矩形包围盒（ImageGrab 只支持矩形截图）
    try:
        # ImageGrab.grab(bbox=(左, 上, 右, 下))
        rect_screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    except Exception as e:
        print(f"错误：截图失败 → {e}")
        return None

    # 步骤4：转换为OpenCV图像，用于透视变换
    rect_cv = cv2.cvtColor(np.array(rect_screenshot), cv2.COLOR_RGB2BGR)
    rect_h, rect_w = rect_cv.shape[:2]

    # 步骤5：计算四边形在矩形截图中的相对坐标（转换为局部坐标，减去包围盒左上角偏移）
    quad_pts_local = quad_pts - np.array([x1, y1], dtype=np.float32)
    quad_pts_local = _order_points(quad_pts_local)  # 排序，保证透视变换正确性

    # 步骤6：定义透视变换的目标坐标（矫正为正矩形）
    dst_pts = np.array([
        [0, 0],  # 左上
        [output_size[0] - 1, 0],  # 右上
        [output_size[0] - 1, output_size[1] - 1],  # 右下
        [0, output_size[1] - 1]   # 左下
    ], dtype=np.float32)

    # 步骤7：执行透视变换，提取四边形区域
    M = cv2.getPerspectiveTransform(quad_pts_local, dst_pts)
    quad_cv = cv2.warpPerspective(rect_cv, M, output_size)

    # 步骤8：转换回PIL.Image格式（保持和ImageGrab一致的返回格式）
    quad_pil = Image.fromarray(cv2.cvtColor(quad_cv, cv2.COLOR_BGR2RGB))

    # 步骤9：可视化结果（可选）
    if visualize:
        # 绘制矩形包围盒和四边形（用于调试）
        rect_show = rect_cv.copy()
        quad_pts_local_int = np.round(quad_pts_local).astype(int)
        cv2.polylines(rect_show, [quad_pts_local_int], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow("矩形包围盒（绿色为四边形）", rect_show)
        cv2.imshow("矫正后的四边形截图", quad_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return quad_pil

# ---------------------- 调用示例 ----------------------
if __name__ == "__main__":
    ROOT_STR = r"C:\Users\ZN191014\Desktop\新建文件夹 (2)"
    folder_name = "quad_screenshot"
    
    # 1. 自定义四边形4个顶点（屏幕坐标，可通过截图工具获取，比如 Snipaste）
    # 格式：[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]，顺序不限（函数内部会自动排序）
    my_quad = [
        (200, 200),   # 点1
        (600, 250),   # 点2
        (580, 600),   # 点3
        (180, 550)    # 点4
    ]

    # 2. 执行四边形截图（矫正为400x400的图像）
    result_img = grab_quadrilateral(
        quad_points=my_quad,
        output_size=(400, 400),
        visualize=True
    )

    # 3. 保存结果（可选）
    if result_img is not None:
        cv2.imwrite(f"{ROOT_STR}\\after\\img\\{folder_name}.png", cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR))
        print("四边形截图已保存为 quad_screenshot.png")