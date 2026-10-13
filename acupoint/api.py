"""FastAPI upload -> ONNX -> annotation -> private history demonstrator."""
import base64
import hmac
import io
import json
import os
import threading
import warnings
from typing import Literal
from datetime import datetime, timedelta, timezone
from pathlib import Path
import jwt
import httpx
from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from .data import read_json
from .inference import OnnxDetector, annotate
from .store import HistoryStore

ROOT = Path(__file__).resolve().parents[1]
REGIONS = {"ear", "face", "hand", "back", "chest", "leg"}


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=256)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    use_llm: bool = False


class ProfileRequest(BaseModel):
    age: int | None = Field(default=None,ge=0,le=120)
    gender: Literal["female","male","unspecified"] = "unspecified"
    preferred_region: Literal["ear","face","hand","back","chest","leg"] = "ear"
    notes: str = Field(default="",max_length=2000)


class UploadRequest(BaseModel):
    content_type: Literal["image/jpeg","image/png","image/webp"] = "image/jpeg"


def create_app(detectors=None, store=None):
    app = FastAPI(title="Acupoint Lab",version="0.1.0")
    app.state.detectors = {} if detectors is None else detectors
    app.state.detector_lock = threading.Lock()
    app.state.store = store or HistoryStore(os.getenv("HISTORY_DB","outputs/history.db"),os.getenv("MONGODB_URI"))
    # A random per-process signing key is allowed locally; configure a shared key for workers.
    app.state.secret = os.getenv("JWT_SECRET") or os.urandom(32).hex()
    security = HTTPBearer(auto_error=False)

    def current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if credentials is None:
            raise HTTPException(401,"Bearer token required")
        try:
            payload = jwt.decode(credentials.credentials,app.state.secret,algorithms=["HS256"],options={"require":["exp","sub"]})
            return payload["sub"]
        except jwt.PyJWTError:
            raise HTTPException(401,"Invalid or expired token")

    def get_detector(region):
        with app.state.detector_lock:
            if region not in app.state.detectors:
                model = os.getenv(f"ACUPOINT_MODEL_{region.upper()}")
                if not model or not Path(model).is_file():
                    raise HTTPException(503,f"No trained ONNX model configured for {region}")
                app.state.detectors[region] = OnnxDetector(model)
            return app.state.detectors[region]

    @app.get("/",response_class=HTMLResponse)
    def ui():
        return (ROOT / "web" / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status":"ok","configured_regions": sorted(set(app.state.detectors) | {
            r for r in REGIONS if os.getenv(f"ACUPOINT_MODEL_{r.upper()}")})}

    @app.post("/api/v1/auth/token")
    def token(request: LoginRequest):
        password = os.getenv("DEMO_PASSWORD")
        if not password:
            raise HTTPException(503,"Set DEMO_PASSWORD to enable demo login")
        if not hmac.compare_digest(request.password.encode(),password.encode()):
            raise HTTPException(401,"Invalid password")
        encoded = jwt.encode({"sub":"demo","exp":datetime.now(timezone.utc)+timedelta(hours=2)},app.state.secret,algorithm="HS256")
        return {"access_token":encoded,"token_type":"bearer","expires_in":7200}

    @app.post("/api/v1/detect")
    async def detect(region: str = Form(...), image: UploadFile = File(...), bbox: str = Form(""), user=Depends(current_user)):
        if region not in REGIONS:
            raise HTTPException(422,"Unsupported region")
        raw = await image.read(8*1024*1024+1)
        if len(raw) > 8*1024*1024:
            raise HTTPException(413,"Image exceeds 8 MB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error",Image.DecompressionBombWarning)
                opened = Image.open(io.BytesIO(raw))
                if opened.width * opened.height > 24_000_000:
                    raise HTTPException(413,"Image exceeds 24 megapixels")
                rgb = ImageOps.exif_transpose(opened).convert("RGB")
            box = json.loads(bbox) if bbox else None
            if box is not None and (not isinstance(box,list) or len(box)!=4 or
                                    any(not isinstance(v,(int,float)) for v in box)):
                raise ValueError("bbox must be [x,y,width,height]")
        except (UnidentifiedImageError,OSError,ValueError,TypeError,Image.DecompressionBombError,Image.DecompressionBombWarning):
            raise HTTPException(422,"Invalid image or bbox")
        detector = await run_in_threadpool(get_detector,region)
        try:
            result = await run_in_threadpool(detector.predict,rgb,box)
        except ValueError as exc:
            raise HTTPException(422,str(exc))
        result["region"] = region
        def finish():
            result_id = app.state.store.add(user,result)
            buffer = io.BytesIO()
            annotate(rgb,result).save(buffer,format="PNG")
            return {"id":result_id,**result,"annotated_image":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()}
        return await run_in_threadpool(finish)

    @app.get("/api/v1/history")
    def history(user=Depends(current_user)):
        return app.state.store.list(user)

    @app.get("/api/v1/profile")
    def get_profile(user=Depends(current_user)):
        return app.state.store.get_profile(user)

    @app.put("/api/v1/profile")
    def set_profile(request:ProfileRequest,user=Depends(current_user)):
        return app.state.store.set_profile(user,request.model_dump())

    @app.post("/api/v1/storage/upload-url")
    def upload_url(request:UploadRequest,user=Depends(current_user)):
        from .storage import signed_upload
        try:
            return signed_upload(user,request.content_type)
        except (RuntimeError,ImportError):
            raise HTTPException(503,"Configure OBS SDK, bucket, server and credentials")

    @app.get("/api/v1/catalog")
    def catalog(q: str = ""):
        entries = read_json(ROOT / "configs" / "catalog.json")
        return [entry for entry in entries if q in entry["name"]]

    @app.post("/api/v1/assistant")
    async def assistant(request: SearchRequest,user=Depends(current_user)):
        catalog_entries = read_json(ROOT / "configs" / "catalog.json")
        matches = [e for e in catalog_entries if e["name"] in request.query or request.query in e["name"]]
        if not request.use_llm:
            return {"mode":"catalog_search","entries":matches,"message":"按穴位名称检索项目资料；未调用大模型。"}
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise HTTPException(503,"DEEPSEEK_API_KEY is not configured")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post("https://api.deepseek.com/chat/completions",headers={"Authorization":f"Bearer {key}"},json={
                    "model":os.getenv("DEEPSEEK_MODEL","deepseek-chat"),"temperature":0.2,
                    "messages":[{"role":"system","content":"你是穴位定位项目的教学资料助手。只解释以下资料中的名称、部位和算法。不要诊断、开处方或给针刺/激光参数。资料："+json.dumps(catalog_entries,ensure_ascii=False)},
                                {"role":"user","content":request.query}]})
                response.raise_for_status()
                answer = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError,KeyError,IndexError,ValueError):
            raise HTTPException(502,"LLM provider unavailable")
        return {"mode":"deepseek","answer":answer,"entries":matches}

    return app


app = create_app()
