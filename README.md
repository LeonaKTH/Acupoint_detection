# Precise Acupuncture Point Location

基于图像的穴位定位项目，包含关键点数据处理、模型训练、ONNX 推理和 FastAPI 接口。

## 功能

- Labelme 点标注转 COCO，标注校验与受试者分组划分。
- CNN + DSNT 关键点回归、注意力融合、Atlas 配准和 Transformer 配准实验。
- PCK、NME、EPE 评估，ONNX 导出与 ROI 坐标还原。
- 图片上传、标注结果、JWT 登录、用户档案和历史记录。
- Web 界面与微信小程序客户端。

耳部基线采用 21 个关键点。面部和手部保留 MediaPipe 映射，胸部、背部和腿部保留 TensorFlow 原型。论文相关模块为实验实现，未完整复现 AIR-Net、RCSAF 或 RT-DEMT。项目不包含数据集和预训练权重，定位结果用于研究。

## 安装

Python 3.10 或以上版本。

```bash
python -m pip install -r requirements.txt
```

原有 MediaPipe / TensorFlow 脚本使用 `requirements-legacy.txt`，建议在 Python 3.11 下运行。OpenMMLab、Mamba 和 OBS SDK 按需单独安装。

## 运行

在项目根目录执行。以下命令使用人工生成的点标记数据验证训练和推理流程。

```bash
python -m scripts.make_synthetic_demo
python -m acupoint.cli train --train outputs/synthetic/train.json --val outputs/synthetic/val.json --images outputs/synthetic/images --epochs 5 --size 64 --output outputs/demo
python -m acupoint.cli export-onnx outputs/demo/best.pt --output outputs/demo/model.onnx
python -m acupoint.cli predict outputs/synthetic/images/synthetic_019.png --model outputs/demo/model.onnx --output outputs/demo/prediction
```

真实数据采用单类别 COCO 关键点格式，名称顺序由 `categories[0].keypoints` 定义。训练按实例 bbox 裁剪；对完整图像推理时，需要提供 `[x, y, width, height]` ROI。图像和标注使用应用 EXIF 方向后的坐标。

```bash
python -m acupoint.cli validate data/train.json --images data/images
python -m acupoint.cli train --train data/train.json --val data/val.json --images data/images --output outputs/model
python -m acupoint.cli evaluate outputs/model/best.pt --annotations data/val.json --images data/images
python -m acupoint.cli convert-labelme data/labels --metadata configs/ear.json --output outputs/coco.json
python -m acupoint.cli atlas examples/atlas.json
```

PCK 和 NME 以 bbox 长边归一化；EPE 使用像素距离。未标注点不参与损失和评估。

## 服务

PowerShell 示例：

```powershell
$env:DEMO_PASSWORD = "your-password"
$env:JWT_SECRET = "your-random-secret-at-least-32-characters"
$env:ACUPOINT_MODEL_EAR = "outputs/demo/model.onnx"
python -m uvicorn acupoint.api:app --host 127.0.0.1 --port 8000
```

访问 `http://127.0.0.1:8000`，接口文档位于 `/docs`。登录接口为 `/api/v1/auth/token`，图片定位接口为 `/api/v1/detect`，上传字段为 `image`、`region` 和可选 `bbox`。

各部位模型通过 `ACUPOINT_MODEL_<REGION>` 设置，需要同名 JSON 元数据。默认使用 SQLite；`MONGODB_URI` 可切换至 MongoDB。DeepSeek 使用 `DEEPSEEK_API_KEY`；OBS 使用 `OBS_ACCESS_KEY_ID`、`OBS_SECRET_ACCESS_KEY`、`OBS_SERVER` 和 `OBS_BUCKET`。演示登录使用共享账号，未配置的部位返回 503。

微信开发者工具导入 `miniprogram/`，修改 `app.js` 中的 API 地址。真机使用 HTTPS 与合法请求域名。

## 目录

```text
acupoint/       数据、模型、配准、推理和 API
configs/        关键点元数据与 OpenMMLab 配置
scripts/        合成数据、级联推理和配准训练
examples/       Atlas 输入示例
web/            Web 客户端
miniprogram/    微信小程序客户端
tests/          数据、坐标、导出与接口测试
third_party/    DSNT 参考源码
ear/ face/ hand/ back/ forebreast/ leg/  原有检测脚本
```

## 测试

```bash
python -m pytest tests -q
```

## 参考

- [DSNT](https://arxiv.org/abs/1801.07372)：`third_party/dsntnn/dsntnn/` 保留 [Aiden Nibali 的原始实现](https://github.com/anibali/dsntnn)及 Apache-2.0 许可证。项目基线使用独立实现的 [0,1] 坐标网格。
- [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)：`scripts/rtmpose_ear.py` 调用 OpenMMLab，需自行准备检测和关键点权重。
- [AIR-Net](https://www.sciencedirect.com/science/article/pii/S0141938224001070)、[RT-DEMT](https://github.com/Sohyu1/RT-DEMT) 与 [RLE](https://github.com/jeffffffli/res-loglikelihood-regression)：算法参考。当前坐标损失为掩码 MSE。

原耳部 OpenMMLab 配置沿用 Tongji Zihao 教程的配置结构，来源信息保留在代码中。
