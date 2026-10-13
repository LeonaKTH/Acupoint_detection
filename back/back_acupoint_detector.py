import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 定义穴位类别
ACUPOINTS = [
    '大椎', '陶道', '身柱', '神道', '灵台', '至阳', '中枢', '筋缩', '脊中', '悬枢', '命门',
    '阳关', '腰阳关', '夹脊', '肺俞', '心俞', '膈俞', '肝俞', '胆俞', '脾俞', '胃俞', '三焦俞',
    '肾俞', '气海俞', '大肠俞', '关元俞', '小肠俞', '膀胱俞', '中膂俞', '白环俞', '上髎', '次髎',
    '中髎', '下髎', '会阳', '秩边', '附分', '魄户', '膏肓', '神堂', '譩譆', '膈关', '魂门',
    '阳纲', '意舍', '胃仓', '肓门', '志室', '胞肓', '秩边'
]


class AcupointDetectionSystem:
    def __init__(self, img_height=512, img_width=512):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = len(ACUPOINTS) + 1  # +1 for background
        self.model = self.build_unet_model()

    def build_unet_model(self):
        """构建U-Net模型用于穴位检测"""
        inputs = Input((self.img_height, self.img_width, 3))

        # 编码器部分
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # 中间部分
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        # 解码器部分
        up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
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
            ModelCheckpoint('best_acupoint_model.h5', save_best_only=True, monitor='val_loss'),
            EarlyStopping(patience=10, monitor='val_loss')
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
        for i, acupoint_name in enumerate(ACUPOINTS, 1):  # 从1开始，0是背景
            coords = np.where(acupoint_map == i)
            if len(coords[0]) > 0:
                # 计算穴位中心点坐标
                y_center = int(np.mean(coords[0]))
                x_center = int(np.mean(coords[1]))
                acupoints_detected[acupoint_name] = (x_center, y_center)

        return acupoints_detected, acupoint_map

    def visualize_results(self, image_path, acupoints_detected, acupoint_map=None):
        """可视化检测结果"""
        # 读取原始图像
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_width, self.img_height))

        # 在图像上标记穴位
        result_img = img.copy()
        for acupoint_name, (x, y) in acupoints_detected.items():
            # 绘制穴位标记
            cv2.circle(result_img, (x, y), 5, (0, 255, 0), -1)
            # 添加穴位名称
            cv2.putText(result_img, acupoint_name, (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 创建图像并显示结果
        plt.figure(figsize=(15, 10))

        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')

        # 显示带有标记的图像
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('检测到的穴位')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('detection_result.jpg')
        plt.show()

        # 如果有穴位分割图，则也显示它
        if acupoint_map is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(acupoint_map, cmap='nipy_spectral')
            plt.title('穴位分割图')
            plt.colorbar(ticks=range(self.num_classes),
                         label='穴位类别')
            plt.savefig('acupoint_segmentation_map.jpg')
            plt.show()

        return result_img


# 数据加载与预处理函数示例
def load_dataset(data_dir, img_height=512, img_width=512, num_classes=len(ACUPOINTS) + 1):
    """加载并预处理数据集"""
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


# 示例：如何使用系统
def main():
    # 初始化系统
    acupoint_system = AcupointDetectionSystem()

    # 训练模式
    # 注意：这需要标注好的数据集
    if False:  # 设置为True以启用训练
        # 加载数据集
        data_dir = 'path/to/dataset'
        images, masks = load_dataset(data_dir)

        # 训练模型
        history = acupoint_system.train(images, masks, epochs=50)

        # 绘制训练历史
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.jpg')
        plt.show()

    # 测试模式 - 使用预训练模型进行穴位检测
    else:
        # 加载预训练模型
        # 实际使用时，应当替换为实际模型路径
        model_path = 'pretrained_acupoint_model.h5'
        try:
            acupoint_system.load_model(model_path)
            print(f"成功加载模型：{model_path}")
        except:
            print("无法加载预训练模型，将使用未训练的模型进行演示")

        # 指定要分析的背部图像
        image_path = 'sample_back_image.jpg'
        print(f"分析图像：{image_path}")

        # 检测穴位
        acupoints_detected, acupoint_map = acupoint_system.detect_acupoints(image_path)

        # 输出检测结果
        print(f"检测到 {len(acupoints_detected)} 个穴位:")
        for acupoint, position in acupoints_detected.items():
            print(f"- {acupoint}: 位置 {position}")

        # 可视化结果
        result_img = acupoint_system.visualize_results(image_path, acupoints_detected, acupoint_map)

        print("检测完成！结果已保存为 'detection_result.jpg' 和 'acupoint_segmentation_map.jpg'")


if __name__ == "__main__":
    main()