function request(path, method='GET', data={}) {
  const app=getApp();
  return new Promise((resolve,reject)=>wx.request({url:app.globalData.apiBase+path,method,data,
    header:{Authorization:'Bearer '+app.globalData.token},
    success:r=>r.statusCode<300?resolve(r.data):reject(new Error(r.data.detail||'请求失败')),fail:reject}));
}
function upload(path,region) {
  const app=getApp();
  return new Promise((resolve,reject)=>wx.uploadFile({url:app.globalData.apiBase+'/api/v1/detect',filePath:path,name:'image',
    formData:{region},header:{Authorization:'Bearer '+app.globalData.token},
    success:r=>{try{const data=JSON.parse(r.data);r.statusCode<300?resolve(data):reject(new Error(data.detail||'上传失败'));}catch(e){reject(e);}},fail:reject}));
}
module.exports={request,upload};
