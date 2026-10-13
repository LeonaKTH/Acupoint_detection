"""Optimize a dense registration experiment on a single atlas/target image pair."""
import argparse
from pathlib import Path
from PIL import Image,ImageOps
import numpy as np
import torch
from acupoint.registration import TransformerRegistrationNet,registration_loss,transfer_points
from acupoint.data import read_json,write_json


def tensor(path,size):
    with Image.open(path) as image:
        image=ImageOps.exif_transpose(image).convert("RGB").resize((size,size))
        return torch.from_numpy(np.asarray(image,dtype=np.float32).transpose(2,0,1).copy()/255)[None]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument("target")
    parser.add_argument("--points",required=True,help="JSON array of normalized template x,y points")
    parser.add_argument("--steps",type=int,default=100)
    parser.add_argument("--size",type=int,default=128)
    parser.add_argument("--output",default="outputs/registration")
    args=parser.parse_args()
    if args.steps<1 or args.size<16:raise ValueError("steps >= 1, size >= 16 required")
    torch.manual_seed(42)
    template,target=tensor(args.template,args.size),tensor(args.target,args.size)
    points=torch.tensor(read_json(args.points),dtype=torch.float32)[None]
    if points.ndim!=3 or points.shape[-1]!=2 or not torch.isfinite(points).all() or ((points<0)|(points>1)).any():
        raise ValueError("Expected finite [K,2] normalized coordinates")
    model=TransformerRegistrationNet()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    losses=[]
    for _ in range(args.steps):
        optimizer.zero_grad()
        flow=model(template,target)
        loss=registration_loss(template,target,flow)
        loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        mapped=transfer_points(points,model(template,target))[0].tolist()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/"result.json",{"target_points":mapped,"losses":losses,
               "scope":"single-pair optimization; not a trained AIR-Net reproduction"})
    torch.save(model.state_dict(),output/"registration.pt")


if __name__=="__main__":main()
