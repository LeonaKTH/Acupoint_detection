"""Generate a deterministic artificial 21-point dataset, without personal photos."""
import argparse
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw


def generate(output, count=20, seed=42):
    if count < 5:
        raise ValueError("At least 5 images required")
    output=Path(output)
    images_dir=output/"images"
    images_dir.mkdir(parents=True,exist_ok=True)
    rng=random.Random(seed)
    names=json.loads((Path(__file__).resolve().parents[1]/"configs/ear.json").read_text(encoding="utf-8"))["names"]
    images,annotations=[],[]
    for i in range(count):
        image=Image.new("RGB",(128,128),"#163d35")
        draw=ImageDraw.Draw(image)
        shift_x,shift_y=rng.randint(-6,6),rng.randint(-6,6)
        keypoints=[]
        for j in range(21):
            x=20+(j%3)*38+shift_x
            y=15+(j//3)*16+shift_y
            draw.ellipse((x-2,y-2,x+2,y+2),fill=(240,180+j*3,80))
            keypoints.extend([x,y,2])
        name=f"synthetic_{i:03d}.png"
        image.save(images_dir/name)
        images.append({"id":i,"file_name":name,"width":128,"height":128,"subject_id":f"synthetic_{i}"})
        annotations.append({"id":i,"image_id":i,"category_id":1,"keypoints":keypoints,
                            "num_keypoints":21,"bbox":[0,0,128,128],"area":16384,"iscrowd":0,"segmentation":[]})
    split=round(count*0.8)
    for name,indices in (("train",range(split)),("val",range(split,count))):
        data={"info":{"description":"Artificial markers; not human anatomy or clinical data"},
              "images":[images[i] for i in indices],"annotations":[annotations[i] for i in indices],
              "categories":[{"id":1,"name":"synthetic_ear","keypoints":names,"skeleton":[]}]}
        (output/f"{name}.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Synthetic dataset: {output.resolve()} ({split} train / {count-split} val)")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="outputs/synthetic")
    parser.add_argument("--count",type=int,default=20)
    args=parser.parse_args()
    generate(args.output,args.count)
