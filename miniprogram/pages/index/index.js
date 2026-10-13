const api=require('../../utils/api');
Page({
  data:{password:'',regions:['耳部','面部','手部','背部'],regionKeys:['ear','face','hand','back'],index:0,image:'',status:'',points:[]},
  onPassword(e){this.setData({password:e.detail.value});},
  onRegion(e){this.setData({index:Number(e.detail.value)});},
  async login(){try{const result=await api.request('/api/v1/auth/token','POST',{password:this.data.password});getApp().globalData.token=result.access_token;this.setData({password:'',status:'已登录'});}catch(e){this.setData({status:e.message});}},
  choose(){wx.chooseMedia({count:1,mediaType:['image'],sourceType:['album','camera'],success:r=>this.setData({image:r.tempFiles[0].tempFilePath,points:[]})});},
  async detect(){if(!this.data.image)return;this.setData({status:'正在定位…'});try{const result=await api.upload(this.data.image,this.data.regionKeys[this.data.index]);
    const file=wx.env.USER_DATA_PATH+'/acupoint-result.png';
    wx.getFileSystemManager().writeFileSync(file,result.annotated_image.split(',')[1],'base64');
    this.setData({image:file,points:result.points,status:'完成，推理耗时 '+result.latency_ms.toFixed(1)+' ms'});
  }catch(e){this.setData({status:e.message});}},
  async history(){try{const records=await api.request('/api/v1/history');this.setData({status:'历史记录 '+records.length+' 条'});}catch(e){this.setData({status:e.message});}}
});
