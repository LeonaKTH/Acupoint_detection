from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import argparse
import matplotlib as mpl # 移到顶部

# --- 字体设置 ---
# 尝试查找系统字体，如果找不到则使用项目目录下的字体文件
try:
    # 优先使用黑体，你可以换成其他字体如 'msyh.ttf' (微软雅黑) 或 'simsun.ttc' (宋体)
    FONT_PATH = "simhei.ttf" # 假设 simhei.ttf 在同级目录下
    # 检查字体文件是否存在
    if not os.path.exists(FONT_PATH):
         # 如果 simhei.ttf 不在，尝试在 Windows 字体目录查找
         font_dir = 'C:/Windows/Fonts'
         if os.path.exists(os.path.join(font_dir, 'simhei.ttf')):
             FONT_PATH = os.path.join(font_dir, 'simhei.ttf')
         elif os.path.exists(os.path.join(font_dir, 'msyh.ttf')): # 尝试微软雅黑
             FONT_PATH = os.path.join(font_dir, 'msyh.ttf')
         elif os.path.exists(os.path.join(font_dir, 'simsun.ttc')): # 尝试宋体
            FONT_PATH = os.path.join(font_dir, 'simsun.ttc')
         else:
             # 在这里可以添加Linux或MacOS的字体路径查找逻辑
             # 如果都找不到，则抛出错误或使用默认（可能不支持中文）
             raise FileNotFoundError("未能找到合适的中文字体文件 (如 simhei.ttf, msyh.ttf)。请将字体文件放到项目目录或指定正确路径。")

    # Matplotlib 全局字体设置
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] + mpl.rcParams['font.sans-serif'] # 优先使用 SimHei 或 微软雅黑
    mpl.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

except Exception as e:
    print(f"字体加载或设置失败: {e}")
    FONT_PATH = None # 标记字体加载失败

# --- Helper Function to Draw Chinese Text using PIL ---
def draw_text_pil(img, text, position, font_path, font_size, color):
    """使用PIL在OpenCV图像上绘制中文"""
    if font_path is None: # 如果字体加载失败，则不绘制文本
        print("警告：字体未加载，无法绘制中文文本。")
        return img
    try:
        font = ImageFont.truetype(font_path, font_size)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color) # PIL颜色是RGB
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"绘制文本时出错: {e}")
        # 如果出错，尝试用OpenCV的默认字体（可能显示问号）作为后备
        try:
            cv2.putText(img, "?", position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30, (color[2], color[1], color[0]), 1) # OpenCV用BGR
        except Exception as e_cv:
             print(f"后备绘制失败: {e_cv}")
        return img


class FaceAcupointDetector:
    ACUPOINT_MAPPINGS = {
        # 额部穴位（根据三焦经定位调整）
        "印堂": 168,  # 两眉连线中点（保持原168）
        "神庭": 10,  # 前发际正中直上0.5寸（改为前额顶部关键点）
        "攒竹": [63, 294],  # 眉头凹陷处（原65→63，更精确的眉内侧点）
        "阳白": [52, 282],  # 眉中点上1寸（保持原点，增加Z轴补偿）
        "头临泣": [67, 296],  # 瞳孔直上入发际0.5寸（改为更靠近发际线的点）

        # 眼部穴位（精确眼眶定位）
        "睛明": [133, 362],  # 内眼角内侧0.1寸（保持正确）
        "瞳子髎": [46, 276],  # 外眼角外0.5寸（原145错误，改为眼角外侧点）
        "承泣": [145, 374],  # 瞳孔直下眶下缘（原159错误，改为下眼眶中点）
        "四白": [112, 341],  # 眶下孔凹陷处（下移关键点）

        # 鼻部穴位（精确鼻翼定位）
        "迎香": [98, 327],  # 鼻翼外缘中点旁0.5寸（原79→98更准确）
        "素髎": 4,  # 鼻尖正中央（原6→4更准确）

        # 口部穴位（精确肌肉附着点）
        "地仓": [61, 291],  # 口角旁开0.4寸（保持正确）
        "颊车": [58, 288],  # 下颌角前上方1寸（保持正确）

        # 新增重要穴位
        "上关": [193, 418],  # 颧弓上缘中央凹陷
        "下关": [174, 397],  # 颧弓下缘凹陷（原位置错误）

        # 耳部穴位（精确耳周解剖）
        "听宫": [234, 454],  # 耳屏前凹陷（保持正确）
        "翳风": [127, 356],  # 耳垂后凹陷（原位置错误）
    }

    ACUPOINT_CATEGORIES = {
        "额部": ["印堂", "神庭", "攒竹", "阳白", "头临泣"],
        "眼部": ["睛明", "瞳子髎", "承泣", "四白"],
        "鼻部": ["迎香", "素髎"],
        "口部": ["地仓", "颊车"],
        "耳部": ["听宫", "翳风", "上关", "下关"]
    }

    CATEGORY_COLORS = {
        "额部": (255, 0, 0),
        "眼部": (0, 128, 0),  # 深绿色更易识别
        "鼻部": (0, 0, 200),  # 深蓝色
        "口部": (0, 255, 255),  # 黄色改为青色
        "耳部": (255, 0, 255)
    }

    def __init__(self, refine_landmarks=True, min_detection_confidence=0.5):
        """
        初始化面部穴位检测器

        Args:
            refine_landmarks: 是否使用细化的面部特征点
            min_detection_confidence: 最小检测置信度
        """
        # 设置MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0)
        )

        # 构建穴位名称到分类的映射
        self.acupoint_to_category = {}
        for category, acupoints in self.ACUPOINT_CATEGORIES.items():
            for acupoint in acupoints:
                self.acupoint_to_category[acupoint] = category


    def _get_acupoint_color(self, acupoint_name: str) -> Tuple[int, int, int]:
        """获取穴位分类对应的颜色"""
        base_name = acupoint_name.split('_')[0] if '_' in acupoint_name else acupoint_name
        category = self.acupoint_to_category.get(base_name, "额部")
        return self.CATEGORY_COLORS[category] # 返回 BGR

    def detect_acupoints(self, image: np.ndarray, show_face_mesh=False, label_size=0.4, circle_size=2) -> Tuple[np.ndarray, Dict]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        results = self.face_mesh.process(image_rgb)
        annotated_image = image.copy()
        acupoints = {}

        # 计算适合PIL的字体大小 (label_size是cv2的比例因子，需要转换)
        pil_font_size = int(label_size * 30) # 经验值，可能需要调整

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                if show_face_mesh:
                    # ... (绘制面部网格不变) ...
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
                    )

                for acupoint_name, landmark_indices in self.ACUPOINT_MAPPINGS.items():
                    if isinstance(landmark_indices, list):
                        left_index, right_index = landmark_indices
                        if left_index < len(landmarks) and right_index < len(landmarks): # 增加索引检查
                            left_point = landmarks[left_index]
                            right_point = landmarks[right_index]
                            left_color_bgr = self._get_acupoint_color(f"{acupoint_name}_左")
                            right_color_bgr = self._get_acupoint_color(f"{acupoint_name}_右")
                            left_color_rgb = (left_color_bgr[2], left_color_bgr[1], left_color_bgr[0]) # 转 RGB 给 PIL
                            right_color_rgb = (right_color_bgr[2], right_color_bgr[1], right_color_bgr[0]) # 转 RGB 给 PIL

                            acupoints[f"{acupoint_name}_左"] = left_point
                            acupoints[f"{acupoint_name}_右"] = right_point

                            cv2.circle(annotated_image, left_point, circle_size, left_color_bgr, -1)
                            # 使用 PIL 绘制文本
                            annotated_image = draw_text_pil(annotated_image, f"{acupoint_name}_左",
                                                            (left_point[0] - 20, left_point[1] - pil_font_size - 5), # 调整位置避免覆盖点
                                                            FONT_PATH, pil_font_size, left_color_rgb)

                            cv2.circle(annotated_image, right_point, circle_size, right_color_bgr, -1)
                            # 使用 PIL 绘制文本
                            annotated_image = draw_text_pil(annotated_image, f"{acupoint_name}_右",
                                                            (right_point[0] - 20, right_point[1] - pil_font_size - 5), # 调整位置
                                                            FONT_PATH, pil_font_size, right_color_rgb)
                        else:
                            print(f"警告: 穴位 '{acupoint_name}' 的索引 {landmark_indices} 超出范围 ({len(landmarks)})")
                    else:
                        if landmark_indices < len(landmarks): # 增加索引检查
                            point = landmarks[landmark_indices]
                            color_bgr = self._get_acupoint_color(acupoint_name)
                            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) # 转 RGB 给 PIL
                            acupoints[acupoint_name] = point

                            cv2.circle(annotated_image, point, circle_size, color_bgr, -1)
                            # 使用 PIL 绘制文本
                            annotated_image = draw_text_pil(annotated_image, acupoint_name,
                                                            (point[0] - 20, point[1] - pil_font_size - 5), # 调整位置
                                                            FONT_PATH, pil_font_size, color_rgb)
                        else:
                             print(f"警告: 穴位 '{acupoint_name}' 的索引 {landmark_indices} 超出范围 ({len(landmarks)})")


        return annotated_image, acupoints

    # --- 修改 detect_specific_acupoints ---
    def detect_specific_acupoints(self, image: np.ndarray, acupoint_list: List[str],
                                  show_face_mesh=False, label_size=0.4, circle_size=2) -> Tuple[np.ndarray, Dict]:
        # (与 detect_acupoints 类似，处理过程和返回值不变，但需要修改内部绘制文本的部分)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        results = self.face_mesh.process(image_rgb)
        annotated_image = image.copy()
        acupoints = {}
        pil_font_size = int(label_size * 30) # 同样转换字体大小

        filtered_mappings = {}
        requested_points = set(acupoint_list) # 使用集合提高查找效率

        for name in acupoint_list:
            base_name = name.split('_')[0] if '_' in name else name
            if base_name in self.ACUPOINT_MAPPINGS:
                filtered_mappings[base_name] = self.ACUPOINT_MAPPINGS[base_name]


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                if show_face_mesh:
                    # ... (绘制面部网格不变) ...
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
                    )


                for acupoint_name, landmark_indices in filtered_mappings.items():
                    if isinstance(landmark_indices, list):
                        left_index, right_index = landmark_indices
                        if left_index < len(landmarks) and right_index < len(landmarks):
                            left_point = landmarks[left_index]
                            right_point = landmarks[right_index]
                            left_name = f"{acupoint_name}_左"
                            right_name = f"{acupoint_name}_右"

                            # 只绘制用户请求的点
                            if left_name in requested_points:
                                left_color_bgr = self._get_acupoint_color(left_name)
                                left_color_rgb = (left_color_bgr[2], left_color_bgr[1], left_color_bgr[0])
                                acupoints[left_name] = left_point
                                cv2.circle(annotated_image, left_point, circle_size, left_color_bgr, -1)
                                annotated_image = draw_text_pil(annotated_image, left_name,
                                                                (left_point[0] - 20, left_point[1] - pil_font_size - 5),
                                                                FONT_PATH, pil_font_size, left_color_rgb)

                            if right_name in requested_points:
                                right_color_bgr = self._get_acupoint_color(right_name)
                                right_color_rgb = (right_color_bgr[2], right_color_bgr[1], right_color_bgr[0])
                                acupoints[right_name] = right_point
                                cv2.circle(annotated_image, right_point, circle_size, right_color_bgr, -1)
                                annotated_image = draw_text_pil(annotated_image, right_name,
                                                                (right_point[0] - 20, right_point[1] - pil_font_size - 5),
                                                                FONT_PATH, pil_font_size, right_color_rgb)
                        else:
                            print(f"警告: 穴位 '{acupoint_name}' 的索引 {landmark_indices} 超出范围 ({len(landmarks)})")

                    else: # 单点
                         if landmark_indices < len(landmarks):
                            point = landmarks[landmark_indices]
                            # 只绘制用户请求的点
                            if acupoint_name in requested_points:
                                color_bgr = self._get_acupoint_color(acupoint_name)
                                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                                acupoints[acupoint_name] = point
                                cv2.circle(annotated_image, point, circle_size, color_bgr, -1)
                                annotated_image = draw_text_pil(annotated_image, acupoint_name,
                                                                (point[0] - 20, point[1] - pil_font_size - 5),
                                                                FONT_PATH, pil_font_size, color_rgb)
                         else:
                             print(f"警告: 穴位 '{acupoint_name}' 的索引 {landmark_indices} 超出范围 ({len(landmarks)})")

        return annotated_image, acupoints


    # --- 修改 create_legend_image ---
    def create_legend_image(self, width=400, height=600):
        """创建图例图像，显示不同分类的穴位颜色"""
        legend_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        y_offset = 30
        category_font_size = 18 # 调整图例字体大小
        acupoint_font_size = 14

        for i, (category, color_bgr) in enumerate(self.CATEGORY_COLORS.items()):
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0]) # 转 RGB

            # 绘制圆点 (OpenCV 可以做)
            cv2.circle(legend_img, (30, y_offset), 10, color_bgr, -1)

            # 使用 PIL 绘制分类名称
            legend_img = draw_text_pil(legend_img, category, (50, y_offset - category_font_size // 2 + 2), # 调整垂直位置
                                       FONT_PATH, category_font_size, (0, 0, 0)) # 黑色文字

            # 列出该类别的穴位
            acupoints_in_category = self.ACUPOINT_CATEGORIES[category]
            # 生成穴位名称（处理左右对称）
            display_names = []
            processed_bases = set()
            for ap_base in acupoints_in_category:
                 if ap_base not in processed_bases:
                     mapping = self.ACUPOINT_MAPPINGS.get(ap_base)
                     if isinstance(mapping, list):
                         display_names.append(f"{ap_base}(左右)")
                     else:
                         display_names.append(ap_base)
                     processed_bases.add(ap_base)


            acupoint_text = ", ".join(display_names[:3]) + ("..." if len(display_names) > 3 else "")
            # 使用 PIL 绘制穴位列表
            legend_img = draw_text_pil(legend_img, acupoint_text, (50, y_offset + 25),
                                       FONT_PATH, acupoint_font_size, (100, 100, 100)) # 灰色文字

            y_offset += 70

        return legend_img

    # --- 修改 visualize_acupoints ---
    @staticmethod
    def visualize_acupoints(image: np.ndarray, acupoints: Dict, categories: Dict = None, figsize=(15, 15)) -> None:
        """
        可视化面部穴位 (确保 Matplotlib 字体设置生效)
        """
        # 字体设置已移到文件顶部
        # import matplotlib as mpl
        # mpl.rcParams['font.family'] = 'Microsoft YaHei' # 或者 'SimHei'
        # mpl.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=figsize)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if categories:
             # ... (绘图逻辑不变，依赖全局字体设置) ...
             # 如果提供了分类信息，使用分类颜色
            colors = {'额部': 'red', '眼部': 'green', '鼻部': 'blue',
                      '口部': 'yellow', '颊面部': 'magenta', '耳部': 'cyan',
                      '颅顶部': 'orange', '下颌部': 'purple'}

            # 将穴位按分类分组
            for category, color in colors.items():
                category_points = []
                category_names = []

                # 获取该分类定义的所有基础穴位名称
                defined_acupoints_in_category = set(categories.get(category, []))

                for name, point in acupoints.items():
                    base_name = name.split('_')[0] if '_' in name else name
                    if base_name in defined_acupoints_in_category: # 检查检测到的点是否属于当前分类
                        category_points.append(point)
                        category_names.append(name) # 使用检测到的完整名称（含左右）

                # 绘制该分类的所有穴位
                if category_points:
                    x_coords = [p[0] for p in category_points]
                    y_coords = [p[1] for p in category_points]
                    plt.scatter(x_coords, y_coords, c=color, s=50, label=category) # 标签是分类名

                    # 添加穴位名称标签
                    for i, name in enumerate(category_names):
                        plt.text(category_points[i][0] + 5, category_points[i][1] + 5, name, fontsize=8, color=color)

            plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # 图例显示分类
        else:
            # ... (简单颜色区分逻辑不变) ...
            middle_color = 'r'
            left_color = 'g'
            right_color = 'b'
            for name, point in acupoints.items():
                if name.endswith("_左"):
                    color = left_color
                elif name.endswith("_右"):
                    color = right_color
                else:
                    color = middle_color
                plt.scatter(point[0], point[1], c=color, s=50)
                plt.text(point[0] + 5, point[1] + 5, name, fontsize=8)


        plt.title("面部穴位定位结果")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def release(self):
        """释放资源"""
        self.face_mesh.close()

# ... (process_image, get_available_acupoints, main 函数保持不变，但会调用修改后的类方法) ...

# --- process_image 函数 (确保调用修改后的方法) ---
def process_image(image_path: str, output_path: str = None, show_face_mesh: bool = False,
                  specific_acupoints: List[str] = None, label_size: float = 0.4, circle_size: int = 2) -> None:
    """
    处理单个图像的面部穴位检测 (使用修改后的类)
    """
    detector = FaceAcupointDetector(refine_landmarks=True)

    # 读取图像，确保能处理中文路径
    try:
        # 使用可以处理非 ASCII 路径的方式读取
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"无法读取图像或图像为空: {image_path}")
    except Exception as e:
         print(f"读取图像时出错: {e}")
         return


    # 检测穴位
    if specific_acupoints:
        annotated_image, acupoints = detector.detect_specific_acupoints(
            image, specific_acupoints, show_face_mesh, label_size, circle_size
        )
    else:
        annotated_image, acupoints = detector.detect_acupoints(
            image, show_face_mesh, label_size, circle_size
        )

    # 生成图例 (现在也使用 PIL 绘制中文)
    legend_img = detector.create_legend_image()

    # 组合图像
    h, w, _ = annotated_image.shape
    legend_h, legend_w, _ = legend_img.shape
    if legend_h != h: # 确保图例高度一致
        aspect_ratio = legend_w / legend_h
        new_legend_h = h
        new_legend_w = int(new_legend_h * aspect_ratio)
        legend_img = cv2.resize(legend_img, (new_legend_w, new_legend_h), interpolation=cv2.INTER_AREA)


    # 创建带图例的完整图像
    combined_img = np.zeros((h, w + legend_img.shape[1], 3), dtype=np.uint8)
    combined_img[:, :w, :] = annotated_image
    combined_img[:, w:, :] = legend_img

    # 保存结果，确保能处理中文路径
    if output_path is None:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        # 确保目录存在
        output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name}_annotated{ext}")

    try:
        # 使用可以处理非 ASCII 路径的方式保存
        is_success, im_buf_arr = cv2.imencode(os.path.splitext(output_path)[1], combined_img)
        if is_success:
            im_buf_arr.tofile(output_path)
            print(f"已保存标注结果到: {output_path}")
        else:
             print(f"保存图像失败: {output_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")


    # 可视化结果 (使用 Matplotlib，依赖全局字体设置)
    detector.visualize_acupoints(annotated_image, acupoints, detector.ACUPOINT_CATEGORIES)

    # 释放资源
    detector.release()


def get_available_acupoints():
    """获取所有可用的穴位名称列表"""
    detector = FaceAcupointDetector()
    all_acupoints = []
    processed_bases = set()

    for acupoint_name, landmark_indices in detector.ACUPOINT_MAPPINGS.items():
        if acupoint_name not in processed_bases:
            if isinstance(landmark_indices, list):
                all_acupoints.append(f"{acupoint_name}_左")
                all_acupoints.append(f"{acupoint_name}_右")
            else:
                all_acupoints.append(acupoint_name)
            processed_bases.add(acupoint_name)

    return sorted(list(all_acupoints)) # 返回排序后的列表


def main():
    parser = argparse.ArgumentParser(description='面部穴位检测工具')
    # 取消注释以允许命令行指定图像
    parser.add_argument('--image', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "head1.jpg"), help='输入图像路径')
    parser.add_argument('--output', type=str, default=None, help='输出图像路径 (默认在输入同目录下加 _annotated 后缀)')
    parser.add_argument('--show-mesh', action='store_true', help='显示面部网格')
    parser.add_argument('--acupoints', type=str, default=None, help='指定需要检测的穴位列表，用逗号分隔 (例如 "印堂,攒竹_左,迎香")')
    parser.add_argument('--list-acupoints', action='store_true', help='列出所有可用的穴位名称')
    parser.add_argument('--label-size', type=float, default=0.4, help='标签文字大小 (影响 OpenCV 绘制比例)')
    parser.add_argument('--circle-size', type=int, default=3, help='穴位圆圈大小') # 稍微增大圆圈

    args = parser.parse_args()

    if args.list_acupoints:
        acupoints = get_available_acupoints()
        print("--------------------")
        print("可用的穴位名称:")
        print("--------------------")
        # 更好看的格式输出
        max_len = max(len(name) for name in acupoints)
        cols = 3 # 每行显示3个
        for i in range(0, len(acupoints), cols):
             row = acupoints[i:i+cols]
             print("  ".join(f"{name:<{max_len}}" for name in row)) # 左对齐并填充空格

        print("--------------------")
        print(f"总计: {len(acupoints)} 个穴位点")
        return

    # 检查字体是否加载成功
    if FONT_PATH is None:
        print("错误：无法继续，因为未能加载中文字体。请确保字体文件可用。")
        return


    specific_acupoints = None
    if args.acupoints:
        specific_acupoints = [x.strip() for x in args.acupoints.split(',') if x.strip()]
        print(f"指定检测穴位: {specific_acupoints}")

    # 设置默认输出路径（如果未指定）
    output_path = args.output
    if output_path is None:
         base_name = os.path.basename(args.image)
         name, ext = os.path.splitext(base_name)
         output_dir = os.path.dirname(args.image)
         output_path = os.path.join(output_dir, f"{name}_annotated{ext}")


    process_image(
        image_path=args.image,
        output_path=output_path,
        show_face_mesh=args.show_mesh,
        specific_acupoints=specific_acupoints,
        label_size=args.label_size,
        circle_size=args.circle_size
    )


if __name__ == "__main__":
    main()
