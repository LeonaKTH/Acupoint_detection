import io
import jwt
from datetime import datetime,timedelta,timezone
from PIL import Image
from fastapi.testclient import TestClient
from acupoint.api import create_app
from acupoint.store import HistoryStore


class Detector:
    def predict(self,image,bbox=None):
        return {"width":image.width,"height":image.height,"backend":"test","latency_ms":1.0,
                "points":[{"id":0,"name":"demo","x":10.,"y":10.}]}


def client(tmp_path,monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD","test-password")
    monkeypatch.setenv("JWT_SECRET","test-secret-with-more-than-thirty-two-characters")
    app=create_app({"ear":Detector()},HistoryStore(tmp_path/"history.db"))
    return TestClient(app)


def test_upload_auth_validation_and_history(tmp_path,monkeypatch):
    c=client(tmp_path,monkeypatch)
    assert c.get('/health').status_code==200
    assert c.get('/').status_code==200
    assert c.get('/api/v1/history').status_code==401
    assert c.post('/api/v1/auth/token',json={'password':'bad'}).status_code==401
    token=c.post('/api/v1/auth/token',json={'password':'test-password'}).json()['access_token']
    headers={'Authorization':'Bearer '+token}
    buffer=io.BytesIO();Image.new('RGB',(20,20)).save(buffer,format='PNG')
    r=c.post('/api/v1/detect',headers=headers,data={'region':'ear'},files={'image':('test.png',buffer.getvalue(),'image/png')})
    assert r.status_code==200 and r.json()['annotated_image'].startswith('data:image/png;base64,')
    assert len(c.get('/api/v1/history',headers=headers).json())==1
    assert c.post('/api/v1/detect',headers=headers,data={'region':'ear'},files={'image':('x',b'bad','image/png')}).status_code==422
    assert c.post('/api/v1/detect',headers=headers,data={'region':'ear','bbox':'[1,2]'},files={'image':('x',buffer.getvalue(),'image/png')}).status_code==422
    assert c.post('/api/v1/detect',headers=headers,data={'region':'back'},files={'image':('x',buffer.getvalue(),'image/png')}).status_code==503


def test_user_history_isolation(tmp_path,monkeypatch):
    c=client(tmp_path,monkeypatch)
    c.app.state.store.add('alice',{'points':[]})
    token=jwt.encode({'sub':'bob','exp':datetime.now(timezone.utc)+timedelta(minutes=1)},c.app.state.secret,algorithm='HS256')
    assert c.get('/api/v1/history',headers={'Authorization':'Bearer '+token}).json()==[]


def test_catalog_and_disabled_llm(tmp_path,monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    c=client(tmp_path,monkeypatch)
    token=c.post('/api/v1/auth/token',json={'password':'test-password'}).json()['access_token']
    headers={'Authorization':'Bearer '+token}
    assert c.get('/api/v1/catalog?q=合谷').json()[0]['name']=='合谷'
    assert c.post('/api/v1/assistant',headers=headers,json={'query':'合谷'}).json()['mode']=='catalog_search'
    assert c.post('/api/v1/assistant',headers=headers,json={'query':'合谷','use_llm':True}).status_code==503


def test_profile_and_disabled_obs(tmp_path,monkeypatch):
    monkeypatch.delenv('OBS_BUCKET',raising=False)
    c=client(tmp_path,monkeypatch)
    token=c.post('/api/v1/auth/token',json={'password':'test-password'}).json()['access_token']
    headers={'Authorization':'Bearer '+token}
    assert c.put('/api/v1/profile',headers=headers,json={'age':22,'preferred_region':'hand'}).status_code==200
    assert c.get('/api/v1/profile',headers=headers).json()['age']==22
    assert c.put('/api/v1/profile',headers=headers,json={'age':200}).status_code==422
    assert c.post('/api/v1/storage/upload-url',headers=headers,json={}).status_code==503


def test_obs_scoped_key_and_signed_headers(monkeypatch):
    from acupoint.storage import signed_upload
    from types import SimpleNamespace
    monkeypatch.setenv('OBS_BUCKET','test-bucket')
    class MockClient:
        def createSignedUrl(self,**kwargs):
            assert kwargs['method']=='PUT' and kwargs['expires']==300
            assert kwargs['bucketName']=='test-bucket'
            assert kwargs['objectKey'].startswith('uploads/') and '../' not in kwargs['objectKey']
            return SimpleNamespace(signedUrl='https://example.test/signed',actualSignedRequestHeaders=kwargs['headers'])
    result=signed_upload('alice','image/png',MockClient())
    assert result['headers']=={'Content-Type':'image/png'}
