"""COCO ROI training/evaluation/export with explicit checkpoint metadata."""
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from .data import read_json, write_json, image_path, validate_coco
from .metrics import keypoint_metrics
from .models import KeypointNet, masked_coordinate_loss


class CocoKeypoints(Dataset):
    def __init__(self, annotation, image_root, size=128, augment=False):
        self.data = read_json(annotation)
        report = validate_coco(self.data, image_root)
        if report["errors"]:
            raise ValueError("; ".join(report["errors"][:10]))
        if len(self.data["categories"]) != 1:
            raise ValueError("One keypoint category required per training run")
        self.names = self.data["categories"][0]["keypoints"]
        self.images = {i["id"]: i for i in self.data["images"]}
        self.annotations = [a for a in self.data["annotations"] if a["num_keypoints"] > 0]
        if not self.annotations:
            raise ValueError("No labeled instances")
        self.root, self.size, self.augment = image_root, size, augment

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        ann = self.annotations[index]
        info = self.images[ann["image_id"]]
        with Image.open(image_path(self.root, info["file_name"])) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        if image.size != (info["width"], info["height"]):
            raise ValueError(f"COCO dimensions disagree with pixels: {info['file_name']}")
        x, y, w, h = ann["bbox"]
        # Use exactly the same clamped integer ROI convention during inference.
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(image.width, int(np.ceil(x+w))), min(image.height, int(np.ceil(y+h)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("ROI outside image")
        image = image.crop((x0, y0, x1, y1)).resize((self.size, self.size))
        if self.augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        points = np.asarray(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
        coordinates = (points[:, :2] - [x0, y0]) / [x1-x0, y1-y0]
        visible = points[:, 2] > 0
        if np.any((coordinates[visible] < 0) | (coordinates[visible] > 1)):
            raise ValueError(f"Labeled points outside ROI: annotation {ann['id']}")
        coordinates[~visible] = 0
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()/255)
        return tensor, torch.tensor(coordinates, dtype=torch.float32), torch.tensor(visible), torch.tensor([x1-x0,y1-y0], dtype=torch.float32)


def check_disjoint(train, val):
    if train.names != val.names:
        raise ValueError("Train/validation keypoint order differs")
    a = {i["file_name"] for i in train.images.values()}
    b = {i["file_name"] for i in val.images.values()}
    if a & b:
        raise ValueError("Train/validation image leakage")
    a = {i["subject_id"] for i in train.images.values() if "subject_id" in i}
    b = {i["subject_id"] for i in val.images.values() if "subject_id" in i}
    if a & b:
        raise ValueError("Train/validation subject leakage")


def evaluate(model, dataset, device="cpu", batch_size=8):
    model.eval()
    predictions, targets, visibility, scales = [], [], [], []
    with torch.inference_mode():
        for image, target, visible, dimensions in DataLoader(dataset, batch_size=batch_size):
            pred = model(image.to(device)).cpu() * dimensions[:, None, :]
            predictions.append(pred.numpy())
            targets.append((target*dimensions[:, None, :]).numpy())
            visibility.append(visible.numpy())
            scales.append(dimensions.amax(1).numpy())
    return keypoint_metrics(np.concatenate(predictions), np.concatenate(targets),
                            np.concatenate(visibility), np.concatenate(scales))


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.size < 16:
        raise ValueError("epochs/batch_size must be positive, size >= 16")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training = CocoKeypoints(args.train, args.images, args.size, augment=True)
    validation = CocoKeypoints(args.val, args.images, args.size)
    check_disjoint(training, validation)
    model = KeypointNet(len(training.names)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0
        for image, target, visible, _ in loader:
            optimizer.zero_grad()
            loss = masked_coordinate_loss(model(image.to(args.device)), target.to(args.device), visible.to(args.device))
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(image)
        metrics = evaluate(model, validation, args.device, args.batch_size)
        row = {"epoch": epoch+1, "train_loss": loss_total/len(training), **metrics}
        history.append(row)
        print(row, flush=True)
        if metrics["nme"] < best:
            best = metrics["nme"]
            torch.save({"state_dict": model.cpu().state_dict(), "names": training.names,
                        "size": args.size, "architecture": "cnn_dsnt_baseline", "seed": args.seed,
                        "metrics": metrics}, output / "best.pt")
            model.to(args.device)
    write_json(output / "history.json", history)
    write_json(output / "run.json", {"arguments": vars(args), "elapsed_seconds": time.perf_counter()-start,
                                     "torch": torch.__version__, "numpy": np.__version__})


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = KeypointNet(len(checkpoint["names"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def export_onnx(checkpoint_path, output):
    import onnx
    import onnxruntime as ort
    model, checkpoint = load_checkpoint(checkpoint_path)
    image = torch.rand(1, 3, checkpoint["size"], checkpoint["size"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, image, str(output), input_names=["image"], output_names=["keypoints"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}, "keypoints": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": image.numpy()})[0]
    with torch.inference_mode():
        expected = model(image).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    write_json(output.with_suffix(".json"), {"names": checkpoint["names"], "size": checkpoint["size"],
               "architecture": checkpoint["architecture"], "coordinates": "normalized_xy_0_1",
               "preprocess": "RGB_NCHW_div255_stretch_ROI", "onnx_max_abs_error": float(abs(actual-expected).max()),
               "metrics": checkpoint["metrics"]})
