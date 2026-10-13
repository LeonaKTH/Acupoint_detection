"""Optional OBS signed PUT URL adapter. No cloud account is contacted at import."""
import hashlib
import os
from uuid import uuid4


def signed_upload(user_id,content_type,client=None):
    types={"image/jpeg":"jpg","image/png":"png","image/webp":"webp"}
    if content_type not in types:
        raise ValueError("Only JPEG, PNG and WebP uploads supported")
    bucket=os.getenv("OBS_BUCKET")
    if not bucket:
        raise RuntimeError("OBS_BUCKET not configured")
    if client is None:
        settings=[os.getenv(key) for key in ("OBS_ACCESS_KEY_ID","OBS_SECRET_ACCESS_KEY","OBS_SERVER")]
        if not all(settings):raise RuntimeError("OBS credentials/server not configured")
        from obs import ObsClient
        client=ObsClient(access_key_id=settings[0],secret_access_key=settings[1],server=settings[2])
    owner=hashlib.sha256(user_id.encode()).hexdigest()[:24]
    key=f"uploads/{owner}/{uuid4().hex}.{types[content_type]}"
    signed=client.createSignedUrl(method="PUT",bucketName=bucket,objectKey=key,expires=300,headers={"Content-Type":content_type})
    return {"object_key":key,"upload_url":signed.signedUrl,"method":"PUT","expires_in":300,
            "headers":signed.actualSignedRequestHeaders}
