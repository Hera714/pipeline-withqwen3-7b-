# coding=utf-8
# depth_estimation_3d.py
import torch
import math
import depth_pro


class DepthPro3DModule:
    def __init__(self):
        print("Loading Depth-Pro 3D Module...")
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def estimate_depth(self, image):
        """
        获取深度矩阵以及 Depth-Pro 预测出的当前图片的焦距 (单位：像素)
        """
        image_tensor = self.transform(image)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            # 当 f_px=None 时，触发 Depth-Pro 独有的内参(焦距)自动估计功能
            prediction = self.model.infer(image_tensor, f_px=None)

        depth_matrix = prediction["depth"].cpu().numpy()
        focal_length_px = prediction["focallength_px"].item()

        return depth_matrix, focal_length_px

    def get_3d_coordinates(self, depth_matrix, point_2d, focal_length_px):
        """
        利用单点坐标、深度和动态预测的焦距，绝对精确地反推 3D 物理坐标 (X, Y, Z) 单位: 米
        基于平视(第一人称)假设：相机在(0,0,0), +X向右(右边物体X更大), +Y向下(低处物体Y更大), +Z向前
        """
        x, y = point_2d
        h, w = depth_matrix.shape

        # 边界保护
        x = min(max(0, int(x)), w - 1)
        y = min(max(0, int(y)), h - 1)

        # 获取 Z 轴深度 (距离镜头的绝对物理距离，单位：米)
        z_val = float(depth_matrix[y, x])

        # 假设图像的几何中心就是相机的主点 (Principal Point)
        cx, cy = w / 2.0, h / 2.0

        # 针孔相机极简反投影公式 (利用真实焦距消除画面长宽比带来的扭曲)
        x_val = (x - cx) * z_val / focal_length_px
        y_val = (y - cy) * z_val / focal_length_px

        return {
            "X_meters": round(x_val, 2),
            "Y_meters": round(y_val, 2),
            "Z_meters": round(z_val, 2)
        }

    def get_3d_bbox_dimensions(self, depth_matrix, bbox_2d, anchor_point_2d, focal_length_px):
        """
        利用 bbox 和避障锚点，计算物体的物理高度和宽度
        """
        x_min, y_min, x_max, y_max = bbox_2d
        ax, ay = anchor_point_2d
        h, w = depth_matrix.shape

        # 1. 提取锚点深度 (Z)
        ax = min(max(0, int(ax)), w - 1)
        ay = min(max(0, int(ay)), h - 1)
        z_val = float(depth_matrix[ay, ax])
        
        if z_val <= 0: return None # 深度异常保护

        cx, cy = w / 2.0, h / 2.0

        # 2. 计算顶部中心点和底部中心点的 Y 物理坐标
        # Y_val = (y - cy) * z / focal
        top_y_meters = (y_min - cy) * z_val / focal_length_px
        bottom_y_meters = (y_max - cy) * z_val / focal_length_px
        
        # 3. 计算左右边缘的 X 物理坐标
        left_x_meters = (x_min - cx) * z_val / focal_length_px
        right_x_meters = (x_max - cx) * z_val / focal_length_px

        # 计算绝对高度和宽度
        physical_height = abs(bottom_y_meters - top_y_meters)
        physical_width = abs(right_x_meters - left_x_meters)

        return {
            "Physical_Height": round(physical_height, 2),
            "Physical_Width": round(physical_width, 2),
            "Top_Y_meters": round(top_y_meters, 2),
            "Bottom_Y_meters": round(bottom_y_meters, 2)
        }