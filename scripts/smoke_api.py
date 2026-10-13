"""Exercise a real ONNX model through FastAPI without starting a public service."""
import argparse
import os
from pathlib import Path
from fastapi.testclient import TestClient
from acupoint.api import create_app
from acupoint.inference import OnnxDetector
from acupoint.store import HistoryStore


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--image",required=True)
    args=parser.parse_args()
    os.environ["DEMO_PASSWORD"]="local-smoke-password"
    app=create_app({"ear":OnnxDetector(args.model)},HistoryStore("outputs/smoke_api.db"))
    with TestClient(app) as client:
        token=client.post('/api/v1/auth/token',json={'password':os.environ['DEMO_PASSWORD']}).json()['access_token']
        with open(args.image,'rb') as image:
            response=client.post('/api/v1/detect',headers={'Authorization':'Bearer '+token},
                                 data={'region':'ear'},files={'image':(Path(args.image).name,image,'image/png')})
        response.raise_for_status()
        result=response.json()
        print({'status':response.status_code,'points':len(result['points']),'backend':result['backend'],'latency_ms':result['latency_ms']})


if __name__=='__main__':main()
