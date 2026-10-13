# 修改后的耳部穴位识别系统，适配COCO格式的Ear210_Dataset
import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO


class EarAcupointDetectorCOCO:
    def __init__(self):
        self.img_size = (224, 224)
        self.model = None
        self.acupoint_names = []  # 将从COCO数据集中获取
        self.num_acupoints = 0  # 将从COCO数据集中获取
        self.coco = None

    def load_coco_dataset(self, annotation_file):
        """加载COCO格式的耳部穴位数据集"""
        self.coco = COCO(annotation_file)

        # 获取所有类别（穴位）
        self.cat_ids = self.coco.getCatIds()  # 保存为类的属性
        categories = self.coco.loadCats(self.cat_ids)
        if len(categories) != 1 or not categories[0].get('keypoints'):
            raise ValueError("需要包含keypoints名称的单类别COCO标注")
        self.acupoint_names = list(categories[0]['keypoints'])
        self.num_acupoints = len(self.acupoint_names)

        print(f"加载了{self.num_acupoints}个穴位类别: {self.acupoint_names}")

        # 获取所有图像ID
        img_ids = self.coco.getImgIds()
        print(f"数据集中包含{len(img_ids)}张图像")

        return self.coco, img_ids

    def prepare_dataset(self, img_ids, img_dir, validation_split=0.2):
        """从COCO数据集准备训练和验证数据"""
        # 确保已经加载了数据集
        if not hasattr(self, 'cat_ids') or self.cat_ids is None:
            raise ValueError("请先调用load_coco_dataset方法加载数据集")

        image_paths = []
        labels = []

        for img_id in img_ids:
            # 获取图像信息
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])

            if not os.path.exists(img_path):
                print(f"警告: 图像文件不存在 {img_path}")
                continue

            # 图像宽高
            width = img_info['width']
            height = img_info['height']

            # 获取该图像的所有标注
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if len(anns) > 1:
                raise ValueError("旧版整图回归每张图片仅支持一个耳部实例；多实例请使用新的ROI训练链路")

            # 初始化穴位坐标标签，默认为-1（表示不存在）
            points = np.ones((self.num_acupoints, 2)) * -1

            for ann in anns:
                category_id = ann['category_id']
                if category_id not in self.cat_ids:
                    raise ValueError("标注类别不在COCO元数据中")

                # 获取关键点坐标
                if 'keypoints' in ann:
                    # COCO格式的关键点是[x, y, visibility]的列表
                    keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                    if len(keypoints) != self.num_acupoints:
                        raise ValueError("标注关键点数量与类别元数据不一致")
                    for i, kp in enumerate(keypoints):
                        if kp[2] > 0:  # 关键点可见
                            # 归一化坐标
                            x, y = kp[0] / width, kp[1] / height
                            points[i] = [x, y]

            # 将点列表展平为一维数组
            flat_points = points.flatten()

            image_paths.append(img_path)
            labels.append(flat_points)

        # 转换为numpy数组
        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # 分割为训练集和验证集
        train_images, val_images, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=validation_split, random_state=42
        )

        return train_images, val_images, train_labels, val_labels

    def build_model(self):
        """构建基于MobileNetV2的耳部穴位检测模型"""
        # 确保已经加载了数据集并设置了num_acupoints
        if self.num_acupoints == 0:
            raise ValueError("请先使用load_coco_dataset加载数据集")

        # 基础模型 - MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.img_size[0], self.img_size[1], 3),
            include_top=False,
            weights='imagenet'
        )

        # 冻结基础模型的层
        for layer in base_model.layers:
            layer.trainable = False

        # 构建完整模型
        x = base_model.output
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)

        # 输出层 - 每个穴位的(x,y)坐标，所以总共是 num_acupoints * 2 个值
        output = Dense(self.num_acupoints * 2, activation='linear')(x)

        # 创建模型
        self.model = Model(inputs=base_model.input, outputs=output)

        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.custom_loss,
            metrics=[self.masked_mae]
        )

        return self.model

    def custom_loss(self, y_true, y_pred):
        """自定义损失函数，只计算有标注的点的损失"""
        # 创建掩码，仅对有效标注点计算损失
        mask = tf.not_equal(y_true, -1)
        # 将无效点的预测值设为与真实值相同，这样它们不会贡献损失
        y_pred_masked = tf.where(mask, y_pred, y_true)
        # 计算有效点的均方误差
        squared_diff = tf.square(y_true - y_pred_masked)
        # 计算有效点的数量
        num_valid = tf.reduce_sum(tf.cast(mask, tf.float32))
        # 返回平均损失
        return tf.reduce_sum(squared_diff) / (num_valid + tf.keras.backend.epsilon())

    def masked_mae(self, y_true, y_pred):
        mask = tf.not_equal(y_true, -1)
        diff = tf.where(mask, tf.abs(y_true-y_pred), tf.zeros_like(y_pred))
        return tf.reduce_sum(diff) / (tf.reduce_sum(tf.cast(mask, tf.float32)) + tf.keras.backend.epsilon())

    def data_generator(self, image_paths, labels, batch_size=32, training=True):
        """数据生成器，用于训练过程中加载和预处理数据"""
        num_samples = len(image_paths)
        if num_samples == 0:
            raise ValueError("数据集不能为空")
        while True:
            indices = np.random.permutation(num_samples) if training else np.arange(num_samples)
            num_batches = int(np.ceil(num_samples / batch_size))

            for i in range(num_batches):
                batch_indices = indices[i * batch_size: min((i + 1) * batch_size, num_samples)]
                batch_images = []
                batch_labels = []

                for idx in batch_indices:
                    # 加载图像
                    img_path = image_paths[idx]
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"无法加载图像: {img_path}")

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)

                    # 数据增强
                    if training and np.random.random() > 0.5:
                        img = self.apply_augmentation(img)

                    # 归一化
                    img = img / 255.0

                    batch_images.append(img)
                    batch_labels.append(labels[idx])

                yield np.array(batch_images), np.array(batch_labels)

    def apply_augmentation(self, image):
        """应用数据增强技术"""
        # 随机亮度调整
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        # 随机对比度调整
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        # 几何增强必须同步变换标签；此处只保留不改变坐标的亮度/对比度增强。

        return image

    def train(self, train_images, train_labels, validation_images, validation_labels,
              epochs=50, batch_size=32, callbacks=None):
        """训练模型"""
        if self.model is None:
            self.build_model()

        # 默认回调函数
        if callbacks is None:
            callbacks = [
                ModelCheckpoint('ear_acupoint_model_best.h5', save_best_only=True, monitor='val_loss'),
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]

        # 创建数据生成器
        train_gen = self.data_generator(train_images, train_labels, batch_size)
        val_gen = self.data_generator(validation_images, validation_labels, batch_size, training=False)

        # 训练模型
        history = self.model.fit(
            train_gen,
            steps_per_epoch=int(np.ceil(len(train_images) / batch_size)),
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=int(np.ceil(len(validation_images) / batch_size)),
            callbacks=callbacks
        )

        return history

    def predict(self, image_path):
        """预测图像中的穴位位置"""
        if self.model is None:
            raise ValueError("模型尚未加载或训练，请先加载或训练模型")

        # 加载并预处理图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")

        original_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.img_size)
        img_normalized = img_resized / 255.0

        # 进行预测
        predictions = self.model.predict(np.expand_dims(img_normalized, axis=0))[0]

        # 将预测结果转换回原始图像尺寸
        h, w, _ = original_img.shape
        predictions = predictions.reshape(-1, 2)
        predictions[:, 0] = predictions[:, 0] * w  # x坐标
        predictions[:, 1] = predictions[:, 1] * h  # y坐标

        return original_img, predictions

    def visualize_predictions(self, image, predictions):
        """可视化预测结果"""
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 绘制每个穴位及其名称
        for i, (x, y) in enumerate(predictions):
            # 只绘制有效的预测点
            if x >= 0 and y >= 0:
                plt.plot(x, y, 'ro', markersize=8)
                plt.text(x + 5, y + 5, self.acupoint_names[i], color='white',
                         bbox=dict(facecolor='red', alpha=0.5))

        plt.title("耳部穴位定位结果")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('ear_acupoint_detection_result.png')
        plt.show()

    def load_model(self, model_path):
        """加载预训练模型"""
        self.model = tf.keras.models.load_model(model_path, compile=False)
        if self.num_acupoints and self.model.output_shape[-1] != self.num_acupoints * 2:
            raise ValueError("旧权重输出维度不匹配21个关键点；修复标签解析后需要重新训练")
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=self.custom_loss, metrics=[self.masked_mae])

    def save_model(self, model_path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型尚未构建，无法保存")
        self.model.save(model_path)

    def evaluate(self, test_images, test_labels):
        """评估模型性能"""
        if self.model is None:
            raise ValueError("模型尚未加载或训练，请先加载或训练模型")

        # 创建测试数据生成器
        test_gen = self.data_generator(test_images, test_labels, batch_size=32, training=False)

        # 评估模型
        loss, mae = self.model.evaluate(test_gen, steps=int(np.ceil(len(test_images) / 32)))
        print(f"测试集上的损失: {loss}")
        print(f"测试集上的平均绝对误差: {mae}")

        return loss, mae


# 使用示例
def train_ear_acupoint_detector():
    """使用Ear210_Dataset_coco数据集训练耳部穴位检测器"""
    # 数据集路径
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    annotation_file = os.path.join(data_root, "train_coco.json")
    img_dir = os.path.join(data_root, "images")

    # 创建检测器
    detector = EarAcupointDetectorCOCO()

    # 加载COCO数据集
    coco, img_ids = detector.load_coco_dataset(annotation_file)

    # 准备训练和验证数据
    train_images, val_images, train_labels, val_labels = detector.prepare_dataset(img_ids, img_dir)

    # 构建模型
    model = detector.build_model()
    print("模型构建完成")
    model.summary()

    # 训练模型
    history = detector.train(train_images, train_labels, val_images, val_labels, epochs=50)

    # 保存模型
    detector.save_model("ear_acupoint_detector_model.h5")

    # 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('轮次')
    plt.legend(['训练', '验证'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['masked_mae'])
    plt.plot(history.history['val_masked_mae'])
    plt.title('平均绝对误差')
    plt.ylabel('MAE')
    plt.xlabel('轮次')
    plt.legend(['训练', '验证'], loc='upper right')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    return detector


# 预测示例
def predict_example(detector, image_path):
    """使用训练好的模型预测图像中的耳部穴位"""
    # 加载模型（如果尚未加载）
    if detector.model is None:
        detector.load_model("ear_acupoint_detector_model.h5")

    # 预测
    image, predictions = detector.predict(image_path)

    # 可视化结果
    detector.visualize_predictions(image, predictions)


if __name__ == "__main__":
    # 训练模型
    detector = train_ear_acupoint_detector()

    # 预测示例
    test_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_image.jpg")
    predict_example(detector, test_image)
