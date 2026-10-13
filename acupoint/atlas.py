"""Multi-template affine landmark registration baseline (not AIR-Net)."""
import numpy as np


def register_atlas(source_landmarks, target_landmarks, source_acupoints):
    source, target, points = [np.asarray(x, float) for x in
                              (source_landmarks, target_landmarks, source_acupoints)]
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("At least 3 paired 2D landmarks required")
    if points.ndim != 2 or points.shape[1] != 2 or not all(np.isfinite(x).all() for x in (source,target,points)):
        raise ValueError("All points must be finite 2D coordinates")
    design = np.c_[source, np.ones(len(source))]
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Collinear template landmarks cannot define an affine transform")
    transform = np.linalg.lstsq(design, target, rcond=None)[0]
    error = float(np.linalg.norm(design @ transform - target, axis=1).mean())
    return np.c_[points, np.ones(len(points))] @ transform, error


def fuse_atlases(templates, target_landmarks):
    registered, errors = zip(*(register_atlas(t["landmarks"], target_landmarks, t["acupoints"])
                               for t in templates))
    registered = np.stack(registered)
    weights = 1 / np.maximum(np.asarray(errors), 1e-6)
    weights /= weights.sum()
    return {"points": np.average(registered, axis=0, weights=weights).tolist(),
            "template_weights": weights.tolist(), "registration_errors": list(errors)}
