import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 定义腿部穴位类别
LEG_ACUPOINTS = [
    # 足太阳膀胱经穴位
    '承扶', '殷门', '浮郄', '委中', '委阳', '合阳', '承筋', '承山', '飞扬', '附阳', '昆仑', '金门', '申脉', '至阴',
    # 足少阳胆经穴位
    '环跳', '风市', '中渎', '膝阳关', '阳陵泉', '阳交', '悬钟', '丘墟', '临泣', '地五会', '侠溪', '足临泣', '足窍阴',
    # 足阳明胃经穴位
    '髀关', '伏兔', '阴市', '梁丘', '犊鼻', '足三里', '上巨虚', '条口', '下巨虚', '丰隆', '解溪', '冲阳', '陷谷',
    '内庭', '厉兑',
    # 足太阴脾经穴位
    '阴陵泉', '漏谷', '地机', '阴辅', '商丘', '三阴交', '太白', '大都', '隐白',
    # 足少阴肾经穴位
    '照海', '然谷', '太溪', '大钟', '水泉', '复溜', '交信', '筑宾', '阴谷',
    # 足厥阴肝经穴位
    '中都', '蠡沟', '中封', '曲泉', '行间', '大敦',
    # 奇经八脉穴位
    '阴跷', '阳跷', '阴维', '阳维',
    # 其他重要腿部穴位
    '膝眼', '鹤顶', '百虫窝', '膝下', '血海'
]


class LegAcupointDetectionSystem:
    def __init__(self, img_height=640, img_width=384):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = len(LEG_ACUPOINTS) + 1  # +1 for background
        self.model = self.build_enhanced_unet_model()

    def build_enhanced_unet_model(self):
        """构建增强版的U-Net模型用于腿部穴位检测"""
        inputs = Input((self.img_height, self.img_width, 3))

        # 编码器部分
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # 中间部分
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        # 解码器部分
        up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        # 注意力机制：让模型更关注穴位所在区域
        attention = Conv2D(1, 1, activation='sigmoid')(conv9)
        conv9 = concatenate([conv9, attention], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

        # 输出层 - 使用softmax为每个像素分配穴位类别的概率
        outputs = Conv2D(self.num_classes, 1, activation='softmax')(conv9)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def train(self, images, masks, validation_split=0.2, epochs=50, batch_size=8):
        """训练模型"""
        # 划分训练集和验证集
        images_train, images_val, masks_train, masks_val = train_test_split(
            images, masks, test_size=validation_split, random_state=42)

        # 设置回调函数
        callbacks = [
            ModelCheckpoint('best_leg_acupoint_model.h5', save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=15, monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6)
        ]

        # 训练模型
        history = self.model.fit(
            images_train, masks_train,
            validation_data=(images_val, masks_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

        return history

    def load_model(self, model_path):
        """加载预训练模型"""
        self.model.load_weights(model_path)

    def preprocess_image(self, image_path):
        """预处理输入图像"""
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = img / 255.0  # 归一化
        return np.expand_dims(img, axis=0)  # 添加批次维度

    def detect_acupoints(self, image_path):
        """检测图像中的穴位"""
        # 预处理输入图像
        preprocessed_img = self.preprocess_image(image_path)

        # 进行预测
        prediction = self.model.predict(preprocessed_img)[0]

        # 获取每个像素的穴位类别
        acupoint_map = np.argmax(prediction, axis=-1)

        # 提取穴位坐标
        acupoints_detected = {}
        for i, acupoint_name in enumerate(LEG_ACUPOINTS, 1):  # 从1开始，0是背景
            coords = np.where(acupoint_map == i)
            if len(coords[0]) > 0:
                # 计算穴位中心点坐标
                y_center = int(np.mean(coords[0]))
                x_center = int(np.mean(coords[1]))
                confidence = float(np.mean(prediction[:, :, i][coords]))  # 计算置信度
                acupoints_detected[acupoint_name] = {
                    'position': (x_center, y_center),
                    'confidence': confidence
                }

        return acupoints_detected, acupoint_map

    def visualize_results(self, image_path, acupoints_detected, acupoint_map=None, confidence_threshold=0.5):
        """可视化检测结果"""
        # 读取原始图像
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_width, self.img_height))

        # 创建不同方向的图像结果
        result_img_front = img.copy()
        result_img_side = img.copy()

        # 按照经络分类穴位
        meridian_colors = {
            '足太阳膀胱经': (255, 0, 0),  # 红色
            '足少阳胆经': (0, 255, 0),  # 绿色
            '足阳明胃经': (0, 0, 255),  # 蓝色
            '足太阴脾经': (255, 255, 0),  # 黄色
            '足少阴肾经': (255, 0, 255),  # 紫色
            '足厥阴肝经': (0, 255, 255),  # 青色
            '奇经八脉': (128, 128, 128),  # 灰色
            '其他': (255, 255, 255)  # 白色
        }

        # 穴位与经络的映射
        acupoint_to_meridian = {}
        for acupoint in LEG_ACUPOINTS:
            if acupoint in ['承扶', '殷门', '浮郄', '委中', '委阳', '合阳', '承筋', '承山',
                            '飞扬', '附阳', '昆仑', '金门', '申脉', '至阴']:
                acupoint_to_meridian[acupoint] = '足太阳膀胱经'
            elif acupoint in ['环跳', '风市', '中渎', '膝阳关', '阳陵泉', '阳交', '悬钟',
                              '丘墟', '临泣', '地五会', '侠溪', '足临泣', '足窍阴']:
                acupoint_to_meridian[acupoint] = '足少阳胆经'
            elif acupoint in ['髀关', '伏兔', '阴市', '梁丘', '犊鼻', '足三里', '上巨虚',
                              '条口', '下巨虚', '丰隆', '解溪', '冲阳', '陷谷', '内庭', '厉兑']:
                acupoint_to_meridian[acupoint] = '足阳明胃经'
            elif acupoint in ['阴陵泉', '漏谷', '地机', '阴辅', '商丘', '三阴交', '太白', '大都', '隐白']:
                acupoint_to_meridian[acupoint] = '足太阴脾经'
            elif acupoint in ['照海', '然谷', '太溪', '大钟', '水泉', '复溜', '交信', '筑宾', '阴谷']:
                acupoint_to_meridian[acupoint] = '足少阴肾经'
            elif acupoint in ['中都', '蠡沟', '中封', '曲泉', '行间', '大敦']:
                acupoint_to_meridian[acupoint] = '足厥阴肝经'
            elif acupoint in ['阴跷', '阳跷', '阴维', '阳维']:
                acupoint_to_meridian[acupoint] = '奇经八脉'
            else:
                acupoint_to_meridian[acupoint] = '其他'

        # 前视图和侧视图穴位分类
        front_view_meridians = ['足阳明胃经', '足太阴脾经', '足厥阴肝经', '其他']
        side_view_meridians = ['足太阳膀胱经', '足少阳胆经', '足少阴肾经', '奇经八脉', '其他']

        # 在图像上标记穴位
        for acupoint_name, data in acupoints_detected.items():
            if data['confidence'] < confidence_threshold:
                continue

            position = data['position']
            x, y = position
            meridian = acupoint_to_meridian.get(acupoint_name, '其他')
            color = meridian_colors[meridian]

            # 决定在哪个视图上显示穴位
            if meridian in front_view_meridians:
                # 绘制穴位标记在前视图
                cv2.circle(result_img_front, (x, y), 5, color, -1)
                # 添加穴位名称
                cv2.putText(result_img_front, acupoint_name, (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # 添加置信度
                cv2.putText(result_img_front, f"{data['confidence']:.2f}", (x + 10, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if meridian in side_view_meridians:
                # 绘制穴位标记在侧视图
                cv2.circle(result_img_side, (x, y), 5, color, -1)
                # 添加穴位名称
                cv2.putText(result_img_side, acupoint_name, (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # 添加置信度
                cv2.putText(result_img_side, f"{data['confidence']:.2f}", (x + 10, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 创建图像并显示结果
        plt.figure(figsize=(18, 12))

        # 显示原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')

        # 显示前视图穴位
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(result_img_front, cv2.COLOR_BGR2RGB))
        plt.title('前视图穴位检测')
        plt.axis('off')

        # 显示侧视图穴位
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result_img_side, cv2.COLOR_BGR2RGB))
        plt.title('侧视图穴位检测')
        plt.axis('off')

        # 添加图例
        handles = []
        labels = []
        for meridian, color in meridian_colors.items():
            color_rgb = (color[2] / 255, color[1] / 255, color[0] / 255)  # BGR转RGB
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color_rgb, markersize=10))
            labels.append(meridian)

        plt.figlegend(handles, labels, loc='lower center', ncol=4,
                      bbox_to_anchor=(0.5, 0.02), frameon=False)

        plt.tight_layout()
        plt.savefig('leg_detection_result.jpg', bbox_inches='tight')
        plt.show()

        # 如果有穴位分割图，则也显示它
        if acupoint_map is not None:
            plt.figure(figsize=(8, 12))
            plt.imshow(acupoint_map, cmap='nipy_spectral')
            plt.title('腿部穴位分割图')
            plt.colorbar(ticks=range(self.num_classes),
                         label='穴位类别')
            plt.savefig('leg_acupoint_segmentation_map.jpg')
            plt.show()

        return result_img_front, result_img_side


# 解剖学特征增强的穴位检测系统
class AnatomyEnhancedLegAcupointSystem(LegAcupointDetectionSystem):
    def __init__(self, img_height=640, img_width=384):
        super().__init__(img_height, img_width)
        self.anatomy_model = self.build_anatomy_model()

    def build_anatomy_model(self):
        """构建解剖学特征识别模型"""
        # 这里应该实现一个识别骨骼、肌肉、关节等解剖学特征的模型
        # 为简化示例，返回None
        return None

    def adjust_acupoints_with_anatomy(self, image, acupoints_detected):
        """使用解剖学特征调整穴位位置"""
        # 这里应该实现使用解剖学标志调整穴位位置的逻辑
        # 例如:
        # 1. 使用膝关节位置调整膝眼、犊鼻等穴位
        # 2. 使用踝关节位置调整昆仑、申脉等穴位
        # 3. 使用胫骨和腓骨的位置调整足三里、阳陵泉等穴位

        # 实际应用中，应该返回调整后的穴位位置
        return acupoints_detected

    def detect_acupoints(self, image_path):
        """结合解剖学特征进行穴位检测"""
        # 调用父类的检测方法
        acupoints_detected, acupoint_map = super().detect_acupoints(image_path)

        # 读取图像
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_width, self.img_height))

        # 使用解剖学特征调整穴位位置
        acupoints_detected = self.adjust_acupoints_with_anatomy(img, acupoints_detected)

        return acupoints_detected, acupoint_map


# 增强型数据加载和增强函数
def load_leg_dataset(data_dir, img_height=640, img_width=384, num_classes=len(LEG_ACUPOINTS) + 1):
    """加载并预处理腿部穴位数据集"""
    images = []
    masks = []

    # 这里应该实现实际数据集的加载逻辑
    # 由于没有实际数据，以下代码仅作为示例框架

    image_files = [f for f in os.listdir(os.path.join(data_dir, 'images'))
                   if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        # 加载图像
        img_path = os.path.join(data_dir, 'images', img_file)
        mask_path = os.path.join(data_dir, 'masks', img_file.replace('.jpg', '.png'))

        # 读取和调整图像大小
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_width, img_height))
        img = img / 255.0  # 归一化

        # 读取和处理掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        # 将掩码转换为one-hot编码
        mask_one_hot = tf.keras.utils.to_categorical(mask, num_classes=num_classes)

        images.append(img)
        masks.append(mask_one_hot)

    return np.array(images), np.array(masks)


def advanced_data_augmentation(images, masks):
    """高级数据增强"""
    augmented_images = []
    augmented_masks = []

    for i in range(len(images)):
        img = images[i]
        mask = masks[i]

        # 原始数据
        augmented_images.append(img)
        augmented_masks.append(mask)

        # 水平翻转 - 对于腿部穴位很有用，可以模拟左腿和右腿
        img_flipped = cv2.flip(img, 1)
        mask_flipped = cv2.flip(mask, 1)
        augmented_images.append(img_flipped)
        augmented_masks.append(mask_flipped)

        # 小角度旋转 (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        height, width = img.shape[:2]
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        img_rotated = cv2.warpAffine(img, M, (width, height))
        mask_rotated = cv2.warpAffine(mask, M, (width, height))
        augmented_images.append(img_rotated)
        augmented_masks.append(mask_rotated)

        # 亮度和对比度变化
        alpha = np.random.uniform(0.8, 1.2)  # 对比度
        beta = np.random.uniform(-10, 10)  # 亮度
        img_adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        augmented_images.append(img_adjusted)
        augmented_masks.append(mask)

        # 缩放和裁剪
        scale = np.random.uniform(0.8, 1.2)
        scaled_height = int(height * scale)
        scaled_width = int(width * scale)
        img_scaled = cv2.resize(img, (scaled_width, scaled_height))
        mask_scaled = cv2.resize(mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

        # 如果缩放后图像比原始大小大，则裁剪中心部分
        if scaled_height > height or scaled_width > width:
            start_y = max(0, (scaled_height - height) // 2)
            start_x = max(0, (scaled_width - width) // 2)
            img_scaled = img_scaled[start_y:start_y + height, start_x:start_x + width]
            mask_scaled = mask_scaled[start_y:start_y + height, start_x:start_x + width]
        # 如果缩放后图像比原始大小小，则填充至原始大小
        else:
            pad_y = (height - scaled_height) // 2
            pad_x = (width - scaled_width) // 2
            img_scaled = np.pad(img_scaled, ((pad_y, height - scaled_height - pad_y),
                                             (pad_x, width - scaled_width - pad_x), (0, 0)),
                                mode='constant')
            mask_scaled = np.pad(mask_scaled, ((pad_y, height - scaled_height - pad_y),
                                               (pad_x, width - scaled_width - pad_x), (0, 0)),
                                 mode='constant')

        augmented_images.append(img_scaled)
        augmented_masks.append(mask_scaled)

        # 添加噪声
        noise = np.random.normal(0, 0.05, img.shape)
        img_noisy = np.clip(img + noise, 0, 1)
        augmented_images.append(img_noisy)
        augmented_masks.append(mask)

    return np.array(augmented_images), np.array(augmented_masks)


# 穴位描述信息和临床应用
LEG_ACUPOINT_INFO = {
    '足三里': {
        '位置': '外膝眼下3寸，胫骨前嵴外一横指',
        '功效': '调理脾胃，扶正培元，疏风化湿',
        '主治': '胃肠疾病，面瘫，中风偏瘫，头痛，全身无力等',
        '刺灸法': '直刺1-1.5寸，可灸'
    },
    '阳陵泉': {
        '位置': '膝下腓骨小头前下方凹陷处',
        '功效': '舒筋活络，疏通经络，利胆退黄',
        '主治': '胆囊炎，肝炎，膝关节痛，下肢无力，腰痛，胸胁胀痛等',
        '刺灸法': '直刺0.5-1寸，可灸'
    },
    '三阴交': {
        '位置': '内踝尖上3寸，胫骨内侧面后缘',
        '功效': '健脾补肝肾，调经养血',
        '主治': '月经不调，痛经，遗尿，不孕，脾胃虚弱，水肿，下肢痿痹等',
        '刺灸法': '直刺1-1.5寸，可灸'
    },
    '昆仑': {
        '位置': '外踝后，跟腱外侧凹陷处',
        '功效': '祛风通络，强腰膝，利水消肿',
        '主治': '腰脊强痛，足跟痛，踝关节扭伤，头痛，癫痫等',
        '刺灸法': '直刺0.5-0.8寸，可灸'
    },
    '委中': {
        '位置': '腘窝横纹中点',
        '功效': '祛风活血，疏经通络，泻热凉血',
        '主治': '腰背痛，膝关节痛，中风，高血压，肛肠疾病等',
        '刺灸法': '直刺0.5-1寸，不宜灸'
    }

}

# 继续添加临床应用模块

# 穴位信息查询系统


class AcupointInformationSystem:
    """穴位信息查询系统"""

    def __init__(self):
        """初始化穴位信息库"""
        # 扩展穴位信息库
        self.acupoint_info = LEG_ACUPOINT_INFO.copy()

        # 添加更多穴位信息
        additional_info = {
            '血海': {
                '位置': '大腿内侧，髂前上棘与髌骨内侧端连线的中点上内方2寸',
                '功效': '调经活血，祛风湿，润燥',
                '主治': '月经不调，痛经，闭经，瘾疹，湿疹，皮肤瘙痒，关节炎',
                '刺灸法': '直刺1-1.5寸，可灸'
            },
            '阴陵泉': {
                '位置': '胫骨内侧髁后下方凹陷处',
                '功效': '健脾利湿，通调下焦',
                '主治': '腹胀，水肿，腹泻，膝关节痛，小便不利，带下，遗尿',
                '刺灸法': '直刺1-1.5寸，可灸'
            },
            '曲泉': {
                '位置': '膝关节屈曲时，膝内侧端凹陷处',
                '功效': '疏肝理气，清热利湿',
                '主治': '膝关节疼痛，下肢痿痹，阴部瘙痒，遗精，阳痿，月经不调',
                '刺灸法': '直刺0.5-1寸，可灸'
            },
            '梁丘': {
                '位置': '髌骨上缘外侧凹陷处',
                '功效': '疏风清热，通经活络',
                '主治': '膝关节肿痛，下肢麻木，胃痛，呕吐，腹泻',
                '刺灸法': '直刺0.5-1寸，可灸'
            },
            '承山': {
                '位置': '腓肠肌肌腹下端与跟腱的连接处',
                '功效': '舒筋活络，强腰膝，通便',
                '主治': '腰背痛，下肢痿痹，小腿抽筋，便秘，痔疮',
                '刺灸法': '直刺0.5-1寸，可灸'
            },
            '太溪': {
                '位置': '内踝后，内踝尖与跟腱之间的凹陷处',
                '功效': '滋补肾阴，补益精血，强壮腰膝',
                '主治': '咽喉干痛，口舌生疮，耳鸣，腰膝酸软，失眠，头晕，男性阳痿，女性不孕',
                '刺灸法': '直刺0.5-0.8寸，可灸'
            },
            '丰隆': {
                '位置': '小腿外侧，外踝尖上8寸，胫骨前嵴外2横指',
                '功效': '化痰止咳，和胃降逆，利水消肿',
                '主治': '咳嗽，气喘，胸闷，恶心，呕吐，腹胀，水肿，痰多',
                '刺灸法': '直刺0.5-1寸，可灸'
            },
            '阳辅': {
                '位置': '外踝上4寸，腓骨前缘',
                '功效': '祛风通络，利胆退热',
                '主治': '头痛，目眩，耳鸣，胸胁疼痛，膝关节痛，下肢痿痹',
                '刺灸法': '直刺0.3-0.5寸，可灸'
            },
            '环跳': {
                '位置': '侧卧位，股骨大转子最高点',
                '功效': '疏风散寒，通经活络',
                '主治': '腰腿痛，下肢痿痹，坐骨神经痛，半身不遂',
                '刺灸法': '直刺1-2寸，可灸'
            },
            '阴市': {
                '位置': '大腿前正中线上，髌骨上缘上6寸',
                '功效': '舒筋活络，疏风散寒',
                '主治': '下肢痿痹，膝股关节疼痛，下肢麻木',
                '刺灸法': '直刺1-1.5寸，可灸'
            }
        }

        # 更新穴位信息库
        self.acupoint_info.update(additional_info)

        # 穴位疾病索引 - 记录哪些穴位适用于特定疾病
        self.disease_index = self._build_disease_index()

    def _build_disease_index(self):
        """构建疾病-穴位索引"""
        disease_index = {}

        # 从穴位信息中提取疾病关键词并建立索引
        for acupoint, info in self.acupoint_info.items():
            if '主治' in info:
                diseases = [d.strip() for d in info['主治'].replace('等', '').split('，')]
                for disease in diseases:
                    if disease not in disease_index:
                        disease_index[disease] = []
                    disease_index[disease].append(acupoint)

        return disease_index

    def get_acupoint_info(self, acupoint_name):
        """获取特定穴位的详细信息"""
        if acupoint_name in self.acupoint_info:
            return self.acupoint_info[acupoint_name]
        return None

    def search_by_disease(self, disease_keyword):
        """根据疾病关键词搜索相关穴位"""
        relevant_acupoints = []

        # 直接匹配疾病索引
        for disease, acupoints in self.disease_index.items():
            if disease_keyword in disease:
                relevant_acupoints.extend(acupoints)

        # 在主治描述中搜索关键词
        for acupoint, info in self.acupoint_info.items():
            if '主治' in info and disease_keyword in info['主治'] and acupoint not in relevant_acupoints:
                relevant_acupoints.append(acupoint)

        return list(set(relevant_acupoints))  # 去重

    def search_by_effect(self, effect_keyword):
        """根据功效关键词搜索相关穴位"""
        relevant_acupoints = []

        for acupoint, info in self.acupoint_info.items():
            if '功效' in info and effect_keyword in info['功效']:
                relevant_acupoints.append(acupoint)

        return relevant_acupoints

    def get_acupoints_by_meridian(self, meridian_name):
        """获取特定经络上的所有穴位"""
        # 经络与穴位的映射
        meridian_mapping = {
            '足太阳膀胱经': ['承扶', '殷门', '浮郄', '委中', '委阳', '合阳', '承筋', '承山', '飞扬', '附阳', '昆仑',
                             '金门', '申脉', '至阴'],
            '足少阳胆经': ['环跳', '风市', '中渎', '膝阳关', '阳陵泉', '阳交', '悬钟', '丘墟', '临泣', '地五会', '侠溪',
                           '足临泣', '足窍阴'],
            '足阳明胃经': ['髀关', '伏兔', '阴市', '梁丘', '犊鼻', '足三里', '上巨虚', '条口', '下巨虚', '丰隆', '解溪',
                           '冲阳', '陷谷', '内庭', '厉兑'],
            '足太阴脾经': ['阴陵泉', '漏谷', '地机', '阴辅', '商丘', '三阴交', '太白', '大都', '隐白'],
            '足少阴肾经': ['照海', '然谷', '太溪', '大钟', '水泉', '复溜', '交信', '筑宾', '阴谷'],
            '足厥阴肝经': ['中都', '蠡沟', '中封', '曲泉', '行间', '大敦']
        }

        if meridian_name in meridian_mapping:
            return meridian_mapping[meridian_name]
        return []


# 穴位治疗方案推荐系统
class AcupointTreatmentSystem:
    """穴位治疗方案推荐系统"""

    def __init__(self, acupoint_info_system):
        """初始化治疗方案推荐系统"""
        self.info_system = acupoint_info_system

        # 预设治疗方案库
        self.treatment_protocols = {
            '腰腿痛': {
                '主要穴位': ['环跳', '委中', '昆仑', '阳陵泉'],
                '配伍穴位': ['肾俞', '太溪', '承山'],
                '治疗原则': '疏通经络，活血化瘀，舒筋止痛',
                '治疗方法': '主要穴位每次选用2-3个，配伍穴位选用1-2个，中强刺激，留针20-30分钟，每周3次，10次为一疗程'
            },
            '膝关节痛': {
                '主要穴位': ['血海', '梁丘', '足三里', '阳陵泉', '阴陵泉'],
                '配伍穴位': ['膝眼', '委中', '曲泉'],
                '治疗原则': '活血化瘀，温经散寒，益气补虚',
                '治疗方法': '主要穴位每次选用3-4个，膝眼穴可加用温针灸，留针30分钟，每周3-4次，每疗程10-12次'
            },
            '下肢麻木': {
                '主要穴位': ['环跳', '阳陵泉', '足三里', '三阴交'],
                '配伍穴位': ['委中', '昆仑', '太溪'],
                '治疗原则': '疏通经络，调和气血，通利关节',
                '治疗方法': '主要穴位必用，配伍穴位选用1-2个，中等刺激，留针30分钟，可配合艾灸，每周3次，每疗程10次'
            },
            '脚踝扭伤': {
                '主要穴位': ['昆仑', '丘墟', '解溪', '申脉'],
                '配伍穴位': ['太溪', '三阴交', '足三里'],
                '治疗原则': '活血化瘀，消肿止痛，舒筋活络',
                '治疗方法': '急性期以点刺拔罐为主，消肿后针刺主要穴位，中强刺激，配合红外线照射，每日1次，5-7次为一疗程'
            },
            '足跟痛': {
                '主要穴位': ['昆仑', '太溪', '承山', '申脉'],
                '配伍穴位': ['涌泉', '足三里', '三阴交'],
                '治疗原则': '舒筋活络，祛风散寒，活血止痛',
                '治疗方法': '主要穴位均用，配伍穴位选用1-2个，可在足底配合梅花针叩刺，主穴强刺激，留针20分钟，每周3次，3周为一疗程'
            }
        }

        # 自定义治疗方案
        self.custom_protocols = []

    def recommend_protocol(self, condition):
        """根据病症推荐治疗方案"""
        # 直接匹配预设方案
        for disease, protocol in self.treatment_protocols.items():
            if condition in disease:
                return {
                    '病症': disease,
                    '方案': protocol
                }

        # 如果没有直接匹配，则通过关键词进行推荐
        # 查找相关穴位
        relevant_acupoints = self.info_system.search_by_disease(condition)

        if not relevant_acupoints:
            return {
                '病症': condition,
                '方案': {
                    '主要穴位': [],
                    '配伍穴位': [],
                    '治疗原则': '无匹配的预设治疗方案，请咨询专业医师',
                    '治疗方法': '无匹配的预设治疗方案，请咨询专业医师'
                }
            }

        # 构建自定义方案
        # 选择前3个作为主要穴位，其余作为配伍穴位
        main_points = relevant_acupoints[:min(3, len(relevant_acupoints))]
        secondary_points = relevant_acupoints[min(3, len(relevant_acupoints)):]

        # 添加一些通用穴位作为配伍
        if '足三里' not in main_points and '足三里' not in secondary_points:
            secondary_points.append('足三里')  # 足三里为通用要穴

        # 根据条件确定治疗原则
        if '痛' in condition:
            principle = '活血化瘀，舒筋止痛'
            method = '主要穴位每次均用，配伍穴位选用1-2个，中强刺激，留针30分钟，可配合温针灸，每周3次，10次为一疗程'
        elif '麻木' in condition or '痹' in condition:
            principle = '疏通经络，调和气血，温经散寒'
            method = '主要穴位每次均用，配伍穴位选用1-2个，中等刺激，留针30分钟，每周3次，8-10次为一疗程'
        elif '肿' in condition:
            principle = '活血化瘀，利水消肿，通络止痛'
            method = '主要穴位均用，配伍穴位选用1-2个，轻至中等刺激，留针20分钟，每日1次，7次为一疗程'
        else:
            principle = '根据病情调整治疗原则'
            method = '请咨询专业医师制定详细治疗方案'

        custom_protocol = {
            '主要穴位': main_points,
            '配伍穴位': secondary_points,
            '治疗原则': principle,
            '治疗方法': method
        }

        return {
            '病症': condition,
            '方案': custom_protocol
        }

    def create_custom_protocol(self, condition, main_points, secondary_points, principle, method):
        """创建自定义治疗方案"""
        custom_protocol = {
            '病症': condition,
            '方案': {
                '主要穴位': main_points,
                '配伍穴位': secondary_points,
                '治疗原则': principle,
                '治疗方法': method
            }
        }

        self.custom_protocols.append(custom_protocol)
        return custom_protocol

    def evaluate_treatment_effect(self, detected_acupoints, disease):
        """评估特定疾病的穴位治疗可能效果"""
        if disease in self.treatment_protocols:
            protocol = self.treatment_protocols[disease]
            main_points = protocol['主要穴位']

            # 计算检测到的主要穴位占比
            detected_main_points = [p for p in main_points if p in detected_acupoints]
            coverage = len(detected_main_points) / len(main_points) if main_points else 0

            if coverage >= 0.8:
                effect = "预期治疗效果良好，关键穴位覆盖率高"
            elif coverage >= 0.5:
                effect = "预期治疗效果中等，部分关键穴位已覆盖"
            else:
                effect = "预期治疗效果有限，关键穴位覆盖不足"

            missing_points = [p for p in main_points if p not in detected_acupoints]

            return {
                '疾病': disease,
                '覆盖率': f"{coverage * 100:.1f}%",
                '已检测主要穴位': detected_main_points,
                '未检测主要穴位': missing_points,
                '预期效果': effect
            }

        return {
            '疾病': disease,
            '评估': "无法评估，没有该疾病的预设治疗方案"
        }


# 临床决策支持系统
class ClinicalDecisionSupportSystem:
    """临床决策支持系统"""

    def __init__(self, acupoint_info_system, treatment_system):
        self.info_system = acupoint_info_system
        self.treatment_system = treatment_system

        # 穴位选择权重系统
        self.acupoint_weights = {
            '足三里': 10,  # 常用要穴，权重最高
            '阳陵泉': 9,
            '三阴交': 9,
            '血海': 8,
            '太溪': 8,
            '昆仑': 7,
            '委中': 7,
            '环跳': 7,
            '阴陵泉': 6,
            '曲泉': 6,
            '承山': 5
        }

        # 特定疾病与穴位的匹配评分标准
        self.disease_point_matching = {
            '腰痛': {
                '主穴': ['环跳', '委中', '肾俞', '腰阳关'],
                '次穴': ['阳陵泉', '承山', '昆仑', '太溪'],
            },
            '膝痛': {
                '主穴': ['血海', '足三里', '阳陵泉', '阴陵泉', '膝眼'],
                '次穴': ['梁丘', '委中', '曲泉', '犊鼻'],
            },
            '踝痛': {
                '主穴': ['昆仑', '丘墟', '申脉', '解溪'],
                '次穴': ['太溪', '绝骨', '悬钟', '三阴交'],
            },
            '下肢麻木': {
                '主穴': ['环跳', '阳陵泉', '足三里', '昆仑'],
                '次穴': ['委中', '承山', '悬钟', '阴陵泉'],
            }
        }

    def analyze_detected_acupoints(self, detected_acupoints, patient_condition=None):
        """分析检测到的穴位与患者病情的关联性"""
        analysis_results = {
            '检测到的穴位': list(detected_acupoints.keys()),
            '穴位总数': len(detected_acupoints),
            '穴位经络分布': self._analyze_meridian_distribution(detected_acupoints),
            '重要穴位覆盖': self._analyze_key_points_coverage(detected_acupoints),
        }

        # 如果提供了患者病情，则进行针对性分析
        if patient_condition:
            analysis_results['病情相关分析'] = self._analyze_condition_relevance(detected_acupoints, patient_condition)
            analysis_results['治疗方案推荐'] = self.treatment_system.recommend_protocol(patient_condition)
            analysis_results['治疗效果评估'] = self._evaluate_treatment_potential(detected_acupoints, patient_condition)

        return analysis_results

    def _analyze_meridian_distribution(self, detected_acupoints):
        """分析检测到的穴位在各经络上的分布"""
        meridian_distribution = {
            '足太阳膀胱经': 0,
            '足少阳胆经': 0,
            '足阳明胃经': 0,
            '足太阴脾经': 0,
            '足少阴肾经': 0,
            '足厥阴肝经': 0,
            '其他': 0
        }

        # 穴位与经络的映射辅助函数
        def get_meridian(acupoint):
            meridian_mapping = {
                '足太阳膀胱经': ['承扶', '殷门', '浮郄', '委中', '委阳', '合阳', '承筋', '承山', '飞扬', '附阳', '昆仑',
                                 '金门', '申脉', '至阴'],
                '足少阳胆经': ['环跳', '风市', '中渎', '膝阳关', '阳陵泉', '阳交', '悬钟', '丘墟', '临泣', '地五会',
                               '侠溪', '足临泣', '足窍阴'],
                '足阳明胃经': ['髀关', '伏兔', '阴市', '梁丘', '犊鼻', '足三里', '上巨虚', '条口', '下巨虚', '丰隆',
                               '解溪', '冲阳', '陷谷', '内庭', '厉兑'],
                '足太阴脾经': ['阴陵泉', '漏谷', '地机', '阴辅', '商丘', '三阴交', '太白', '大都', '隐白'],
                '足少阴肾经': ['照海', '然谷', '太溪', '大钟', '水泉', '复溜', '交信', '筑宾', '阴谷'],
                '足厥阴肝经': ['中都', '蠡沟', '中封', '曲泉', '行间', '大敦']
            }

            for meridian, points in meridian_mapping.items():
                if acupoint in points:
                    return meridian
            return '其他'

        # 统计各经络穴位数量
        for acupoint in detected_acupoints:
            meridian = get_meridian(acupoint)
            meridian_distribution[meridian] += 1

        # 移除计数为0的经络
        return {k: v for k, v in meridian_distribution.items() if v > 0}

    def _analyze_key_points_coverage(self, detected_acupoints):
        """分析关键穴位的覆盖情况"""
        key_points = ['足三里', '阳陵泉', '三阴交', '委中', '昆仑', '太溪', '环跳', '血海']
        covered_key_points = [point for point in key_points if point in detected_acupoints]

        return {
            '关键穴位总数': len(key_points),
            '已覆盖关键穴位': covered_key_points,
            '覆盖率': f"{len(covered_key_points) / len(key_points) * 100:.1f}%"
        }

    def _analyze_condition_relevance(self, detected_acupoints, condition):
        """分析检测到的穴位与特定病情的相关性"""
        # 获取与病情相关的穴位
        relevant_acupoints = self.info_system.search_by_disease(condition)

        # 计算检测到的穴位中有多少是与病情相关的
        detected_relevant = [point for point in detected_acupoints if point in relevant_acupoints]

        # 计算重要穴位的覆盖情况
        if condition in self.disease_point_matching:
            matching = self.disease_point_matching[condition]
            main_points = matching['主穴']
            secondary_points = matching['次穴']

            detected_main = [point for point in main_points if point in detected_acupoints]
            detected_secondary = [point for point in secondary_points if point in detected_acupoints]

            main_coverage = len(detected_main) / len(main_points) if main_points else 0
            secondary_coverage = len(detected_secondary) / len(secondary_points) if secondary_points else 0

            # 计算相关性得分 (主穴权重0.7，次穴权重0.3)
            relevance_score = main_coverage * 0.7 + secondary_coverage * 0.3

            # 相关性评级
            if relevance_score >= 0.8:
                relevance_rating = "高度相关"
            elif relevance_score >= 0.5:
                relevance_rating = "中度相关"
            elif relevance_score >= 0.3:
                relevance_rating = "低度相关"
            else:
                relevance_rating = "相关性不足"

            return {
                '相关穴位': relevant_acupoints,
                '已检测相关穴位': detected_relevant,
                '主穴覆盖': f"{main_coverage * 100:.1f}%",
                '次穴覆盖': f"{secondary_coverage * 100:.1f}%",
                '相关性得分': f"{relevance_score * 100:.1f}%",
                '相关性评级': relevance_rating
            }

        # 如果没有预设的匹配标准，则简单计算相关性
        coverage = len(detected_relevant) / len(relevant_acupoints) if relevant_acupoints else 0

        if coverage >= 0.7:
            relevance_rating = "高度相关"
        elif coverage >= 0.4:
            relevance_rating = "中度相关"
        elif coverage >= 0.2:
            relevance_rating = "低度相关"
        else:
            relevance_rating = "相关性不足"

        return {
            '相关穴位': relevant_acupoints,
            '已检测相关穴位': detected_relevant,
            '覆盖率': f"{coverage * 100:.1f}%",
            '相关性评级': relevance_rating
        }

    def _evaluate_treatment_potential(self, detected_acupoints, condition):
        """评估基于检测到的穴位进行治疗的潜力"""
        # 先获取治疗方案
        treatment_rec = self.treatment_system.recommend_protocol(condition)
        protocol = treatment_rec['方案']

        # 检查主要穴位的覆盖情况
        main_points = protocol['主要穴位']
        covered_main = [point for point in main_points if point in detected_acupoints]
        main_coverage = len(covered_main) / len(main_points) if main_points else 0

        # 检查配伍穴位的覆盖情况
        secondary_points = protocol['配伍穴位']
        covered_secondary = [point for point in secondary_points if point in detected_acupoints]
        secondary_coverage = len(covered_secondary) / len(secondary_points) if secondary_points else 0

        # 计算综合治疗潜力得分
        treatment_score = main_coverage * 0.8 + secondary_coverage * 0.2

        # 评估治疗潜力
        if treatment_score >= 0.8:
            potential_rating = "治疗潜力优秀"
            recommendation = "可直接基于检测结果进行穴位治疗"
        elif treatment_score >= 0.6:
            potential_rating = "治疗潜力良好"
            recommendation = "建议补充检测部分关键穴位后进行治疗"
        elif treatment_score >= 0.4:
            potential_rating = "治疗潜力一般"
            recommendation = "建议进一步完善穴位检测，关注主要穴位"
        else:
            potential_rating = "治疗潜力有限"
            recommendation = "建议咨询专业医师，完善评估后制定方案"

        return {
            '主要穴位覆盖率': main_coverage,
            '配伍穴位覆盖率': secondary_coverage,
            '综合得分': treatment_score,
            '评级': potential_rating,
            '建议': recommendation,
        }
