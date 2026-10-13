import numpy as np
import torch
from PIL import Image
from acupoint.models import KeypointNet,dsnt,masked_coordinate_loss
from acupoint.training import export_onnx,CocoKeypoints,check_disjoint
from acupoint.inference import OnnxDetector
from acupoint.data import write_json


def test_dsnt_center_corner_and_gradients():
    uniform=torch.zeros(1,1,3,3,requires_grad=True)
    coordinates=dsnt(uniform)
    torch.testing.assert_close(coordinates,torch.tensor([[[0.5,0.5]]]))
    coordinates.sum().backward()
    assert uniform.grad.abs().sum()>0
    corner=torch.full((1,1,3,3),-50.0);corner[0,0,0,2]=50
    torch.testing.assert_close(dsnt(corner),torch.tensor([[[1.,0.]]]))


def test_masked_loss_gradient():
    pred=torch.tensor([[[1.,2.],[30.,40.]]],requires_grad=True)
    loss=masked_coordinate_loss(pred,torch.zeros_like(pred),torch.tensor([[True,False]]))
    assert loss==5
    loss.backward()
    assert pred.grad[0,1].abs().sum()==0


def test_onnx_parity_and_roi_roundtrip(tmp_path):
    torch.manual_seed(1)
    model=KeypointNet(2).eval()
    checkpoint={"state_dict":model.state_dict(),"names":["a","b"],"size":32,
                "architecture":"cnn_dsnt_baseline","metrics":{}}
    torch.save(checkpoint,tmp_path/"model.pt")
    export_onnx(tmp_path/"model.pt",tmp_path/"model.onnx")
    detector=OnnxDetector(tmp_path/"model.onnx")
    image=Image.new("RGB",(100,80))
    result=detector.predict(image,[20,10,40,30])
    with torch.inference_mode():expected=model(torch.zeros(1,3,32,32))[0].numpy()*[40,30]+[20,10]
    np.testing.assert_allclose([[p['x'],p['y']] for p in result['points']],expected,atol=1e-4)


def test_roi_exif_and_visibility(tmp_path):
    exif=Image.Exif();exif[274]=6
    Image.new("RGB",(40,20)).save(tmp_path/"image.jpg",exif=exif)
    data={"images":[{"id":1,"file_name":"image.jpg","width":20,"height":40}],
          "categories":[{"id":1,"keypoints":["a","b"]}],
          "annotations":[{"id":1,"image_id":1,"category_id":1,"keypoints":[10,20,2,0,0,0],
                          "bbox":[0,0,20,40],"num_keypoints":1}]}
    write_json(tmp_path/"coco.json",data)
    dataset=CocoKeypoints(tmp_path/"coco.json",tmp_path,size=32)
    image,points,visible,dimensions=dataset[0]
    assert image.shape==(3,32,32)
    torch.testing.assert_close(points[0],torch.tensor([0.5,0.5]))
    assert visible.tolist()==[True,False]


def test_registration_flow_direction_and_backward():
    from acupoint.registration import TransformerRegistrationNet,transfer_points,warp_target_to_template,registration_loss
    flow=torch.zeros(1,2,32,32);flow[:,0]=0.2;flow[:,1]=-0.4
    points=torch.tensor([[[0.3,0.6]]])
    torch.testing.assert_close(transfer_points(points,flow),torch.tensor([[[0.4,0.4]]]))
    image=torch.rand(1,3,32,32)
    torch.testing.assert_close(warp_target_to_template(image,torch.zeros_like(flow)),image,atol=1e-5,rtol=1e-5)
    model=TransformerRegistrationNet(channels=16,layers=1)
    predicted=model(image,torch.rand_like(image))
    loss=registration_loss(image,torch.rand_like(image),predicted)
    loss.backward()
    assert model.flow_head.weight.grad.abs().sum()>0
