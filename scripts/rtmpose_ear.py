"""Actual RTMDet -> RTMPose cascade via OpenMMLab, using your trained weights."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",help="Image, video, folder, or webcam")
    parser.add_argument("--det-weights",required=True)
    parser.add_argument("--pose-weights",required=True)
    parser.add_argument("--output",default="outputs/rtmpose")
    parser.add_argument("--device",default="cpu")
    args = parser.parse_args()
    from mmpose.apis import MMPoseInferencer
    root = Path(__file__).resolve().parents[1]
    configs = root / "ear/data/mmpose_Config_Examples_RTMpose_RTMDet"
    inferencer = MMPoseInferencer(pose2d=str(configs / "rtmpose-s-ear.py"),pose2d_weights=args.pose_weights,
                                 det_model=str(configs / "rtmdet_tiny_ear.py"),det_weights=args.det_weights,
                                 det_cat_ids=[0],device=args.device)
    for result in inferencer(args.input,out_dir=args.output):
        print(json.dumps(result["predictions"],ensure_ascii=False))


if __name__ == "__main__":
    main()
