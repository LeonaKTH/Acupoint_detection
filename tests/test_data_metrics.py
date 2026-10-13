import copy
import numpy as np
import pytest
from PIL import Image
from acupoint.data import validate_coco,split_coco,labelme_to_coco,write_json,image_path
from acupoint.metrics import keypoint_metrics
from acupoint.atlas import register_atlas,fuse_atlases
from acupoint.hand import map_hand_landmarks


def sample_coco():
    return {"images":[{"id":i,"file_name":f'{i}.png',"width":100,"height":100,"subject_id":i//2} for i in range(10)],
            "categories":[{"id":1,"name":"demo","keypoints":["a","b"],"skeleton":[[1,2]]}],
            "annotations":[{"id":i,"image_id":i,"category_id":1,"keypoints":[10,20,2,0,0,0],
                            "num_keypoints":1,"bbox":[0,0,50,50]} for i in range(10)]}


def test_coco_invalid_and_split_subjects():
    data=sample_coco()
    assert not validate_coco(data)["errors"]
    train,val=split_coco(data,seed=42)
    assert len(train["images"])==8 and len(val["images"])==2
    assert not {i["subject_id"] for i in train["images"]}&{i["subject_id"] for i in val["images"]}
    assert split_coco(data,seed=42)==(train,val)
    invalid=copy.deepcopy(data)
    invalid["annotations"][0]["keypoints"][0]=101
    assert validate_coco(invalid)["errors"]
    invalid=copy.deepcopy(data)
    invalid["categories"][0]["skeleton"]=[[0,1]]
    assert validate_coco(invalid)["errors"]


def test_path_escape(tmp_path):
    with pytest.raises(ValueError):
        image_path(tmp_path,"../private.jpg")


def test_labelme_missing_points_and_occlusion(tmp_path):
    data={"imageWidth":100,"imageHeight":100,"imagePath":"sample.png","shapes":[
        {"label":"a","shape_type":"point","points":[[10,20]],"flags":{"occluded":True}}]}
    write_json(tmp_path/"sample.json",data)
    result=labelme_to_coco(tmp_path,["a","b"])
    assert result["annotations"][0]["keypoints"]==[10,20,1,0,0,0]
    data["shapes"].append(dict(data["shapes"][0]))
    write_json(tmp_path/"sample.json",data)
    with pytest.raises(ValueError):labelme_to_coco(tmp_path,["a","b"])


def test_exif_size_validation(tmp_path):
    exif=Image.Exif();exif[274]=6
    Image.new("RGB",(100,50)).save(tmp_path/"photo.jpg",exif=exif)
    data=sample_coco();data["images"]=[{"id":0,"file_name":"photo.jpg","width":50,"height":100}]
    data["annotations"]=data["annotations"][:1]
    assert validate_coco(data,tmp_path)["exif_oriented_images"]==1
    assert not validate_coco(data,tmp_path)["errors"]


def test_metrics_masked_and_known_distance():
    target=np.zeros((1,2,2));pred=np.array([[[3,4],[999,999]]])
    metrics=keypoint_metrics(pred,target,[[2,0]],[10],threshold=0.5)
    assert metrics["epe_px"]==5 and metrics["nme"]==0.5
    assert metrics["pck@0.5"]==1 and metrics["failure_rate@0.1"]==1
    with pytest.raises(ValueError):keypoint_metrics(pred,target,[[0,0]],[10])


def test_affine_atlas_and_multi_template():
    source=np.array([[0,0],[1,0],[0,1],[1,1]])
    target=source*2+[4,5]
    mapped,error=register_atlas(source,target,[[0.5,0.5]])
    np.testing.assert_allclose(mapped,[[5,6]],atol=1e-7)
    assert error<1e-6
    result=fuse_atlases([{"landmarks":source,"acupoints":[[0.5,0.5]]}]*2,target)
    np.testing.assert_allclose(result["template_weights"],[0.5,0.5])
    with pytest.raises(ValueError):register_atlas([[0,0],[1,0],[2,0]],[[0,0],[1,0],[2,0]],[[0,0]])


def test_hand_rotation_equivariance():
    landmarks=np.arange(42).reshape(21,2)
    rotated=landmarks@np.array([[0,-1],[1,0]])+[100,200]
    original=map_hand_landmarks(landmarks,view="back")
    actual=map_hand_landmarks(rotated,view="back")
    for name,value in original.items():
        np.testing.assert_allclose(actual[name],np.array(value)@np.array([[0,-1],[1,0]])+[100,200])
    with pytest.raises(ValueError):map_hand_landmarks([])
