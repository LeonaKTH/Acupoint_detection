"""Reusable explicit-view landmark mapping derived from the original hand demo."""
import numpy as np


def map_hand_landmarks(landmarks, view="back"):
    points = np.asarray(landmarks,float)
    if points.shape != (21,2) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite 2D hand landmarks")
    if view not in {"palm","back"}:
        raise ValueError("view must be palm or back")
    result = {f"十宣{i+1}": points[index] for i,index in enumerate((4,8,12,16,20))}
    if view == "palm":
        result.update({"鱼际":points[1],"劳宫":(3*points[9]+points[0])/4,
                       "少府":(3*points[17]+points[0])/4,"中冲":(2*points[12]+points[11])/3})
        result.update({f"四缝{i+1}":points[index] for i,index in enumerate((6,10,14,18))})
    else:
        result.update({"中泉":(3*points[0]+points[1])/4,"大骨空":points[3],
                       "少商":(points[3]+points[4])/2,
                       "三间":((3*points[5]+points[0])/4+points[2])/2,
                       "合谷":((points[5]+points[0])/2+points[2])/2,
                       "后溪":points[17],"中魁":points[10],"小骨空":points[18],
                       "腰痛点1":(points[5]+points[0])/2,"腰痛点2":(points[13]+points[0])/2})
        result.update({f"八邪{i+1}":(points[a]+points[b])/2
                       for i,(a,b) in enumerate(((2,5),(5,9),(9,13),(13,17)))})
    return {name:coordinate.tolist() for name,coordinate in result.items()}


def detect_hands(image, view="back"):
    from PIL import ImageOps
    import mediapipe as mp
    if not hasattr(mp,"solutions"):
        raise RuntimeError("Use the documented MediaPipe 0.10.21 / Python 3.11 legacy environment")
    image = ImageOps.exif_transpose(image).convert("RGB")
    with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=2,min_detection_confidence=0.5) as hands:
        results = hands.process(np.asarray(image))
    detections = []
    for hand in results.multi_hand_landmarks or []:
        landmarks = [[p.x*image.width,p.y*image.height] for p in hand.landmark]
        mapped = map_hand_landmarks(landmarks,view)
        detections.append({"view":view,"width":image.width,"height":image.height,
                           "backend":"mediapipe_geometric_baseline",
                           "points":[{"id":i,"name":name,"x":xy[0],"y":xy[1]}
                                     for i,(name,xy) in enumerate(mapped.items())]})
    return detections
