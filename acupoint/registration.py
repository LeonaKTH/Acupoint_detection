"""Trainable CNN/Transformer dense registration experiment, inspired by atlas methods.

Flow is defined on template pixels and points template -> target, in [-1,1] units.
This implementation is independent and is not the AIR-Net paper architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F


def coordinate_grid(height,width,device,dtype):
    ys=torch.linspace(-1,1,height,device=device,dtype=dtype)
    xs=torch.linspace(-1,1,width,device=device,dtype=dtype)
    yy,xx=torch.meshgrid(ys,xs,indexing="ij")
    return torch.stack((xx,yy),-1)


class TransformerRegistrationNet(nn.Module):
    def __init__(self,channels=32,heads=4,layers=2,max_displacement=0.25):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,channels,3,stride=2,padding=1),nn.GELU(),
                                   nn.Conv2d(channels,channels,3,stride=2,padding=1),nn.GELU(),
                                   nn.Conv2d(channels,channels,3,stride=2,padding=1),nn.GELU())
        self.pair_projection=nn.Linear(channels*2,channels)
        self.position_projection=nn.Linear(2,channels)
        self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(channels,heads,
                                              dim_feedforward=channels*4,dropout=0.0,batch_first=True),layers)
        self.flow_head=nn.Linear(channels,2)
        # Begin at the identity warp. Nonzero gradients reach the flow head immediately.
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        self.max_displacement=max_displacement

    def forward(self,template,target):
        a,b=self.encoder(template),self.encoder(target)
        batch,channels,height,width=a.shape
        tokens=torch.cat((a,b),1).flatten(2).transpose(1,2)
        position=coordinate_grid(height,width,a.device,a.dtype).reshape(1,-1,2)
        tokens=self.pair_projection(tokens)+self.position_projection(position)
        flow=self.flow_head(self.transformer(tokens)).tanh()*self.max_displacement
        flow=flow.transpose(1,2).reshape(batch,2,height,width)
        return F.interpolate(flow,size=template.shape[-2:],mode="bilinear",align_corners=True)


def warp_target_to_template(target,flow):
    batch,_,height,width=flow.shape
    grid=coordinate_grid(height,width,flow.device,flow.dtype)[None]+flow.permute(0,2,3,1)
    return F.grid_sample(target,grid,mode="bilinear",padding_mode="border",align_corners=True)


def transfer_points(points,flow):
    """Map [B,K,2] template points in [0,1] to target points in [0,1]."""
    samples=F.grid_sample(flow,(points*2-1).unsqueeze(2),mode="bilinear",padding_mode="border",align_corners=True)
    return points+samples.squeeze(-1).transpose(1,2)/2


def registration_loss(template,target,flow,smoothness=0.01):
    photo=F.mse_loss(warp_target_to_template(target,flow),template)
    dx=flow[:,:,:,1:]-flow[:,:,:,:-1]
    dy=flow[:,:,1:,:]-flow[:,:,:-1,:]
    return photo+smoothness*(dx.square().mean()+dy.square().mean())
