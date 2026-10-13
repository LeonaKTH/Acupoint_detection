"""ONNX ROI inference, Unicode labels and latency benchmarking."""
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .data import read_json


class OnnxDetector:
    def __init__(self, model_path, providers=None):
        import onnxruntime as ort
        self.metadata = read_json(Path(model_path).with_suffix(".json"))
        if self.metadata.get("coordinates") != "normalized_xy_0_1":
            raise ValueError("Model must export normalized_xy_0_1 with a metadata sidecar")
        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image, bbox=None):
        start = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        if bbox is None:
            x0, y0, x1, y1 = 0, 0, image.width, image.height
        else:
            x, y, w, h = bbox
            if not np.isfinite(bbox).all() or w <= 0 or h <= 0:
                raise ValueError("Invalid bbox")
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(image.width, int(np.ceil(x+w))), min(image.height, int(np.ceil(y+h)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Empty ROI")
        size = self.metadata["size"]
        roi = image.crop((x0,y0,x1,y1)).resize((size,size))
        tensor = np.asarray(roi, dtype=np.float32).transpose(2,0,1)[None]/255
        coordinates = self.session.run(None, {self.input_name: tensor})[0][0]
        if coordinates.shape != (len(self.metadata["names"]), 2) or not np.isfinite(coordinates).all():
            raise ValueError("Model output disagrees with metadata")
        coordinates = coordinates * [x1-x0,y1-y0] + [x0,y0]
        return {"width": image.width, "height": image.height, "bbox": [x0,y0,x1-x0,y1-y0],
                "backend": self.metadata["architecture"], "latency_ms": (time.perf_counter()-start)*1000,
                "points": [{"id": i, "name": name, "x": float(x), "y": float(y)}
                           for i,(name,(x,y)) in enumerate(zip(self.metadata["names"], coordinates))]}


def annotate(image, result):
    image = ImageOps.exif_transpose(image).convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    candidates = [os.getenv("ACUPOINT_FONT", ""), "C:/Windows/Fonts/msyh.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    font = None
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            font = ImageFont.truetype(candidate, max(12, image.width//80))
            break
    for point in result["points"]:
        x,y = point["x"],point["y"]
        r = max(3,image.width//300)
        draw.ellipse((x-r,y-r,x+r,y+r),fill="#ff6b35",outline="white")
        label = point["name"] if font else str(point["id"])
        draw.text((x+r+2,y),label,fill="#ff6b35",font=font)
    return image


def benchmark(detector, image, bbox=None, runs=50, warmup=5):
    if runs < 1 or warmup < 0:
        raise ValueError("runs >= 1, warmup >= 0 required")
    for _ in range(warmup):
        detector.predict(image, bbox)
    timings = [detector.predict(image,bbox)["latency_ms"] for _ in range(runs)]
    return {"runs": runs, "warmup": warmup, "mean_ms": float(np.mean(timings)),
            "p50_ms": float(np.percentile(timings,50)), "p95_ms": float(np.percentile(timings,95)),
            "fps": float(1000/np.mean(timings)), "scope": "ROI preprocess + ONNX + decode; excludes upload"}
