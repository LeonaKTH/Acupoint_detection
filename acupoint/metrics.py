"""Pixel EPE, bbox-normalized NME/PCK and per-instance failure rate."""
import numpy as np


def keypoint_metrics(prediction, target, visibility, scales, threshold=0.2):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    mask = np.asarray(visibility) > 0
    scales = np.asarray(scales, float)
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("Expected matching [N,K,2] predictions and targets")
    if mask.shape != target.shape[:2] or scales.shape != (len(target),):
        raise ValueError("Expected [N,K] visibility and [N] normalization scales")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0) or not 0 < threshold <= 1:
        raise ValueError("Scales must be finite/positive; threshold in (0,1]")
    if not np.isfinite(prediction[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Visible coordinates must be finite")
    distances = np.linalg.norm(prediction - target, axis=-1)
    normalized = distances / scales[:, None]
    if not mask.any():
        raise ValueError("No labeled keypoints to evaluate")
    counts = mask.sum(1)
    sample_nme = np.where(mask, normalized, 0).sum(1) / np.maximum(counts, 1)
    return {"labeled_keypoints": int(mask.sum()), "evaluated_instances": int((counts > 0).sum()),
            "epe_px": float(distances[mask].mean()), "nme": float(normalized[mask].mean()),
            f"pck@{threshold:g}": float((normalized[mask] <= threshold).mean()),
            "failure_rate@0.1": float((sample_nme[counts > 0] > 0.1).mean()),
            "normalization": "max(bbox_width,bbox_height)"}
