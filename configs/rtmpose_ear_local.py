# Overlay the existing MMPose config, retaining the original RTMPose architecture.
from pathlib import Path
_base_ = ['../ear/data/mmpose_Config_Examples_RTMpose_RTMDet/rtmpose-s-ear.py']
data_root = str(Path(__file__).resolve().parents[1] / 'ear/data') + '/'
train_dataloader = dict(dataset=dict(data_root=data_root, ann_file='train_coco.json',data_prefix=dict(img='images/')))
val_dataloader = dict(dataset=dict(data_root=data_root,ann_file='val_coco.json',data_prefix=dict(img='images/')))
test_dataloader = val_dataloader
val_evaluator = [dict(type='CocoMetric',ann_file=data_root+'val_coco.json'),dict(type='PCKAccuracy',thr=0.2),dict(type='NME',norm_mode='keypoint_distance',keypoint_indices=[1,19])]
test_evaluator = val_evaluator
