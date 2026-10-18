"""Compact GRU/LSTM and Gaussian GRU on identically eligible CDM histories."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader,TensorDataset
sys.path.insert(0,'src')
from conjunctions.data import load_raw
from sklearn.metrics import mean_absolute_error,mean_squared_error
from scipy.stats import norm
torch.set_num_threads(4);torch.manual_seed(42);rng=np.random.default_rng(42)
s=pd.read_parquet('outputs/snapshots.parquet');d=load_raw();O=Path('outputs/tables')
cols=['risk','miss_distance','t_sigma_r','t_sigma_t','t_sigma_n','c_sigma_r','c_sigma_t','c_sigma_n','time_to_tca']
d['risk']=d.risk.clip(lower=-15);d['miss_distance']=np.log1p(d.miss_distance.clip(lower=0))
for c in cols[2:8]:d[c]=np.log1p(d[c].clip(lower=0))
d['gap_days']=d.groupby('event_id').time_to_tca.diff().abs().fillna(0)
cols+=['gap_days'];groups={i:g for i,g in d.groupby('event_id',sort=False)}
class Net(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.rnn=(nn.LSTM if kind=='lstm' else nn.GRU)(len(cols)*2,24,batch_first=True);self.head=nn.Linear(24,2 if kind=='prob_gru' else 1)
 def forward(self,x,length):
  _,h=self.rnn(pack_padded_sequence(x,length.cpu(),batch_first=True,enforce_sorted=False));h=h[0] if self.kind=='lstm' else h
  return self.head(h[-1])
def arrays(a,h,median,mean,std):
 xs=[];ls=[]
 for row in a.itertuples():
  z=groups[row.event_id];z=z[z.time_to_tca>=h/24][cols].to_numpy(float)
  mask=np.isnan(z).astype(float);z=np.where(np.isnan(z),median,z);z=(z-mean)/std;z=np.concatenate([z,mask],axis=1)
  xs.append(z);ls.append(len(z))
 x=np.zeros((len(xs),max(ls),len(cols)*2),dtype='float32')
 for i,z in enumerate(xs):x[i,:len(z)]=z
 return torch.tensor(x),torch.tensor(ls),torch.tensor(a.target_risk_capped.to_numpy(),dtype=torch.float32)
def predict(m,z):
 m.eval();out=[]
 with torch.no_grad():
  for x,l,y in DataLoader(z,batch_size=256):out.append(m(x,l).numpy())
 return np.concatenate(out)
rows=[];preds=[]
for h,a in s.groupby('horizon_h'):
 tr=a[a.partition=='train'];va=a[a.partition=='validation'];ca=a[a.partition=='calibration'];te=a[a.partition=='test']
 train_cdms=d[d.event_id.isin(tr.event_id)&(d.time_to_tca>=h/24)][cols].to_numpy(float)
 median=np.nanmedian(train_cdms,axis=0);mean=np.nanmean(train_cdms,axis=0);std=np.nanstd(train_cdms,axis=0);std=np.where(std<1e-6,1,std)
 datasets={k:TensorDataset(*arrays(v,h,median,mean,std)) for k,v in [('train',tr),('validation',va),('calibration',ca),('test',te)]}
 for kind in ['gru','lstm','prob_gru']:
  m=Net(kind);opt=torch.optim.Adam(m.parameters(),lr=.005,weight_decay=.0001);best=np.inf;state=None;patience=0
  for epoch in range(10):
   m.train()
   for x,l,y in DataLoader(datasets['train'],batch_size=128,shuffle=True):
    out=m(x,l);mu=out[:,0]
    if kind=='prob_gru':sd=out[:,1].clamp(-3,3).exp();loss=(.5*((y-mu)/sd)**2+torch.log(sd)).mean()
    else:loss=((y-mu)**2).mean()
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
   v=predict(m,datasets['validation']);vy=va.target_risk_capped.to_numpy();score=mean_absolute_error(vy,v[:,0])
   if score<best-0.002:best=score;state={k:t.clone() for k,t in m.state_dict().items()};patience=0
   else:patience+=1
   if patience>=3:break
  m.load_state_dict(state);pc=predict(m,datasets['calibration']);pe=predict(m,datasets['test']);y=te.target_risk_capped.to_numpy();yc=ca.target_risk_capped.to_numpy()
  if kind=='prob_gru':
   sd=np.exp(np.clip(pe[:,1],-3,3));sc=np.exp(np.clip(pc[:,1],-3,3));scale=np.quantile(abs(yc-pc[:,0])/sc,.9)/norm.ppf(.95);width=norm.ppf(.95)*sd*scale
   nll=float(np.mean(norm.logpdf(y,loc=pe[:,0],scale=np.maximum(sd*scale,.01))*-1))
  else:
   width=np.quantile(abs(yc-pc[:,0]),np.ceil((len(ca)+1)*.9)/len(ca),method='higher');width=np.repeat(width,len(y));nll=np.nan
  rows.append({'horizon_h':h,'model':kind,'n_test':len(te),'validation_mae':best,'mae':mean_absolute_error(y,pe[:,0]),'rmse':np.sqrt(mean_squared_error(y,pe[:,0])),'interval_coverage90':np.mean((y>=pe[:,0]-width)&(y<=pe[:,0]+width)),'interval_width90':float(np.mean(2*width)),'gaussian_nll':nll,'epochs':epoch+1})
  preds.append(pd.DataFrame({'event_id':te.event_id,'horizon_h':h,'model':kind,'y':y,'prediction':pe[:,0],'low':pe[:,0]-width,'high':pe[:,0]+width}))
  torch.save({'state_dict':m.state_dict(),'columns':cols,'median':median,'mean':mean,'std':std},f'outputs/models/{kind}_{h}.pt')
res=pd.DataFrame(rows);res.to_csv(O/'sequence_metrics.csv',index=False);pd.concat(preds).to_csv(O/'sequence_predictions.csv',index=False);print(res.to_string(index=False))
