"""COCO keypoints validation, deterministic group splits and Labelme conversion."""
import copy
import json
import math
import random
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def image_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes image root: {name}")
    return path


def validate_coco(data, image_root=None):
    errors = []
    oriented_images = 0
    images, categories = {}, {}
    for key, target in (("images", images), ("categories", categories)):
        for item in data.get(key, []):
            if item["id"] in target:
                errors.append(f"Duplicate {key} id: {item['id']}")
            target[item["id"]] = item
    for cat in categories.values():
        names = cat.get("keypoints", [])
        if not names or len(names) != len(set(names)):
            errors.append(f"Invalid keypoint names in category {cat['id']}")
        for a, b in cat.get("skeleton", []):
            if not (1 <= a <= len(names) and 1 <= b <= len(names)):
                errors.append(f"Skeleton must use 1-based COCO indices: {cat['id']}")
    for img in images.values():
        if img.get("width", 0) <= 0 or img.get("height", 0) <= 0:
            errors.append(f"Invalid dimensions: image {img['id']}")
        if image_root is not None:
            try:
                path = image_path(image_root, img["file_name"])
                if not path.is_file():
                    errors.append(f"Missing image: {img['file_name']}")
                else:
                    from PIL import Image
                    with Image.open(path) as pixels:
                        orientation = pixels.getexif().get(274,1)
                        oriented_images += int(orientation != 1)
                        size = pixels.size[::-1] if orientation in (5,6,7,8) else pixels.size
                        if size != (img["width"],img["height"]):
                            errors.append(f"EXIF-oriented dimensions mismatch: {img['file_name']}")
            except (ValueError,OSError) as exc:
                errors.append(str(exc))
    ids = set()
    for ann in data.get("annotations", []):
        aid = ann["id"]
        if aid in ids:
            errors.append(f"Duplicate annotation id: {aid}")
        ids.add(aid)
        img, cat = images.get(ann["image_id"]), categories.get(ann["category_id"])
        if img is None or cat is None:
            errors.append(f"Dangling annotation: {aid}")
            continue
        points = ann.get("keypoints", [])
        if len(points) != 3 * len(cat.get("keypoints", [])):
            errors.append(f"Wrong keypoint count: annotation {aid}")
            continue
        visible = 0
        for x, y, v in zip(points[0::3], points[1::3], points[2::3]):
            if v not in (0, 1, 2) or not math.isfinite(x) or not math.isfinite(y):
                errors.append(f"Invalid keypoint: annotation {aid}")
            if v > 0:
                visible += 1
                if not (0 <= x < img["width"] and 0 <= y < img["height"]):
                    errors.append(f"Keypoint outside image: annotation {aid}")
        if ann.get("num_keypoints") != visible:
            errors.append(f"num_keypoints mismatch: annotation {aid}")
        box = ann.get("bbox", [])
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= 0 or box[3] <= 0:
            errors.append(f"Invalid bbox: annotation {aid}")
    if not images or not categories:
        errors.append("Images and categories must not be empty")
    return {"images": len(images), "annotations": len(ids), "categories": len(categories),
            "exif_oriented_images": oriented_images, "errors": errors}


def split_coco(data, val_ratio=0.2, seed=42, group_field="subject_id"):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    report = validate_coco(data)
    if report["errors"]:
        raise ValueError("Invalid COCO: " + "; ".join(report["errors"][:5]))
    groups = {}
    for img in data["images"]:
        group = ("subject", img[group_field]) if group_field in img else ("image", img["id"])
        groups.setdefault(group, []).append(img["id"])
    if len(groups) < 2:
        raise ValueError("Need at least two independent image/subject groups")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    count = max(1, min(len(keys) - 1, round(len(keys) * val_ratio)))
    val_ids = {iid for key in keys[:count] for iid in groups[key]}
    outputs = []
    for is_val in (False, True):
        result = copy.deepcopy(data)
        result["images"] = [i for i in result["images"] if (i["id"] in val_ids) == is_val]
        image_ids = {i["id"] for i in result["images"]}
        result["annotations"] = [a for a in result["annotations"] if a["image_id"] in image_ids]
        outputs.append(result)
    return tuple(outputs)


def labelme_to_coco(folder, names, category="acupoint"):
    """One image/one instance. Multiple group_ids are rejected rather than merged."""
    if not names or len(names) != len(set(names)):
        raise ValueError("Keypoint names must be nonempty and unique")
    images, annotations = [], []
    files = sorted(Path(folder).glob("*.json"))
    if not files:
        raise ValueError("No Labelme JSON files found")
    for iid, file in enumerate(files):
        data = read_json(file)
        width, height = data["imageWidth"], data["imageHeight"]
        points, groups = {}, set()
        for shape in data["shapes"]:
            if shape["shape_type"] != "point":
                continue
            label = shape["label"]
            if label not in names or label in points:
                raise ValueError(f"Unknown/duplicate keypoint {label} in {file.name}")
            groups.add(shape.get("group_id"))
            x, y = shape["points"][0]
            visibility = 1 if shape.get("flags", {}).get("occluded", False) else 2
            points[label] = [x, y, visibility]
        if len(groups) > 1 or not points:
            raise ValueError(f"Expected exactly one labeled instance: {file.name}")
        coords = list(points.values())
        x0, y0 = min(p[0] for p in coords), min(p[1] for p in coords)
        x1, y1 = max(p[0] for p in coords), max(p[1] for p in coords)
        # Expand degenerate boxes and keep the ROI inside the image.
        x0, y0 = max(0, x0 - 5), max(0, y0 - 5)
        x1, y1 = min(width, x1 + 5), min(height, y1 + 5)
        images.append({"id": iid, "file_name": Path(data["imagePath"]).name, "width": width, "height": height})
        annotations.append({"id": iid, "image_id": iid, "category_id": 1,
                            "keypoints": [v for name in names for v in points.get(name, [0, 0, 0])],
                            "num_keypoints": len(points), "bbox": [x0, y0, x1-x0, y1-y0],
                            "area": (x1-x0)*(y1-y0), "iscrowd": 0, "segmentation": []})
    result = {"images": images, "annotations": annotations,
              "categories": [{"id": 1, "name": category, "keypoints": names, "skeleton": []}]}
    errors = validate_coco(result)["errors"]
    if errors:
        raise ValueError("; ".join(errors[:5]))
    return result
