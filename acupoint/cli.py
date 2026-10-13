import argparse
import json
from .data import read_json, write_json, validate_coco, split_coco, labelme_to_coco


def main(argv=None):
    parser = argparse.ArgumentParser(description="穴位定位数据、训练与部署工具")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("annotations")
    p.add_argument("--images")
    p = sub.add_parser("split")
    p.add_argument("annotations")
    p.add_argument("--output", default="outputs/split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p = sub.add_parser("convert-labelme")
    p.add_argument("folder")
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--output", default="outputs/ear_baseline")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("export-onnx")
    p.add_argument("checkpoint")
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("checkpoint")
    p.add_argument("--annotations", required=True)
    p.add_argument("--images", required=True)
    p = sub.add_parser("predict")
    p.add_argument("image")
    p.add_argument("--model", required=True)
    p.add_argument("--bbox", nargs=4, type=float)
    p.add_argument("--output", default="outputs/prediction")
    p = sub.add_parser("benchmark")
    p.add_argument("image")
    p.add_argument("--model", required=True)
    p.add_argument("--bbox", nargs=4, type=float)
    p.add_argument("--runs", type=int, default=50)
    p = sub.add_parser("atlas")
    p.add_argument("input",help="JSON with templates and target_landmarks")
    p.add_argument("--output",default="outputs/atlas.json")
    p = sub.add_parser("hand")
    p.add_argument("image")
    p.add_argument("--view",choices=["palm","back"],default="back")
    p.add_argument("--output",default="outputs/hand")
    args = parser.parse_args(argv)
    if args.command == "validate":
        report = validate_coco(read_json(args.annotations),args.images)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if report["errors"]:
            raise SystemExit(1)
    elif args.command == "split":
        train, val = split_coco(read_json(args.annotations),args.val_ratio,args.seed)
        write_json(f"{args.output}/train.json",train)
        write_json(f"{args.output}/val.json",val)
    elif args.command == "convert-labelme":
        metadata = read_json(args.metadata)
        write_json(args.output,labelme_to_coco(args.folder,metadata["names"],metadata.get("region","acupoint")))
    elif args.command == "train":
        from .training import train
        train(args)
    elif args.command == "export-onnx":
        from .training import export_onnx
        export_onnx(args.checkpoint,args.output)
    elif args.command == "evaluate":
        from .training import load_checkpoint,CocoKeypoints,evaluate
        model, checkpoint = load_checkpoint(args.checkpoint)
        dataset = CocoKeypoints(args.annotations,args.images,checkpoint["size"])
        if dataset.names != checkpoint["names"]:
            raise ValueError("Checkpoint keypoint order differs from annotations")
        print(json.dumps(evaluate(model,dataset),indent=2))
    elif args.command == "atlas":
        from .atlas import fuse_atlases
        data = read_json(args.input)
        write_json(args.output,fuse_atlases(data["templates"],data["target_landmarks"]))
    elif args.command == "hand":
        from PIL import Image
        from .hand import detect_hands
        from .inference import annotate
        with Image.open(args.image) as image:
            detections = detect_hands(image,args.view)
            write_json(args.output+".json",detections)
            for result in detections:
                image = annotate(image,result)
            image.save(args.output+".png")
    else:
        from PIL import Image
        from .inference import OnnxDetector,annotate,benchmark
        detector = OnnxDetector(args.model)
        with Image.open(args.image) as original:
            image = original.convert("RGB")
        if args.command == "benchmark":
            print(json.dumps(benchmark(detector,image,args.bbox,args.runs),indent=2))
        else:
            result = detector.predict(image,args.bbox)
            write_json(args.output+".json",result)
            annotate(image,result).save(args.output+".png")


if __name__ == "__main__":
    main()
