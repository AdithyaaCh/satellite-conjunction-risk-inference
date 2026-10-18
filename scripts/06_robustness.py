"""Sparse histories, mission transfer, official selection bias, and event bootstrap."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import mean_absolute_error,precision_score,recall_score
from xgboost import XGBRegressor
s=pd.read_parquet('outputs/snapshots.parquet');p=pd.read_csv('outputs/tables/test_predictions.csv');O=Path('outputs/tables');rng=np.random.default_rng(42)
# Measured test subsets, with denominator shown.
# Paired horizon comparison on events observed at all four cutoffs.
common=set.intersection(*(set(p[(p.horizon_h==h)&(p.model=='xgboost')].event_id) for h in [48,72,96,120]))
common_horizon=[]
for h in [120,96,72,48]:
 t=p[(p.horizon_h==h)&(p.model=='xgboost')&p.event_id.isin(common)]
 common_horizon.append({'horizon_h':h,'n':len(t),'mae':float(mean_absolute_error(t.y,t.prediction)),'interval_coverage90':float(((t.y>=t.low)&(t.y<=t.high)).mean())})
a=s[(s.horizon_h==72)&(s.partition=='test')].set_index('event_id');q=p[(p.horizon_h==72)&(p.model=='xgboost')].set_index('event_id')
z=a.join(q[['prediction']],how='inner');sub={}
for label,mask in [('sparse_1to3',z.n_cdm<=3),('dense_4plus',z.n_cdm>=4),('low_LEO',z.regime=='low LEO'),('middle_LEO',z.regime=='middle LEO'),('upper_LEO',z.regime=='upper LEO')]:
 t=z[mask];sub[label]={'n':len(t),'mae':float(mean_absolute_error(t.target_risk_capped,t.prediction)) if len(t) else None,'high_risk_1e5':int(t.high_risk_1e5.sum())}
# Fixed model, mission-held-out transfer: mission identity has no unique satellite information.
b=s[s.horizon_h==72].copy();b['risk_now']=b.risk.clip(lower=-15);b['log_miss']=np.log1p(b.miss_distance);b['log_sigma']=np.log1p(b.t_sigma_norm+b.c_sigma_norm)
feat=['risk_now','log_miss','log_sigma','n_cdm','t_h_per'];mission=b.mission_id.value_counts().index[0]
train=b[(b.mission_id!=mission)&b.partition.isin(['train','validation'])];test=b[(b.mission_id==mission)&b.partition.eq('test')]
model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10)).fit(train[feat],train.target_risk_capped)
transfer={'held_out_mission':int(mission),'n':len(test),'mae':float(mean_absolute_error(test.target_risk_capped,model.predict(test[feat]))),'note':'Mission is only a proxy for repeated spacecraft; exact identity and dates are unavailable.'}
# Competition data: distinct local event ID spaces; use official test only after fitting official train.
root=Path('Collision Avoidance Challenge - Dataset/kelvins_competition_data')
tr=pd.read_csv(root/'train_data.zip',compression='zip');te=pd.read_csv(root/'test_data.csv');truth=pd.read_csv(root/'test_data_private.csv',usecols=['event_id','true_risk'])
def snap(d,training):
 d=d.sort_values(['event_id','time_to_tca'],ascending=[True,False]);last=d.groupby('event_id',sort=False).tail(1).set_index('event_id')
 x=d[d.time_to_tca>=2].groupby('event_id',sort=False).tail(1).set_index('event_id')
 if training:x=x[last.loc[x.index,'time_to_tca'].to_numpy()<=1].copy();x['y']=last.loc[x.index,'risk'].clip(lower=-15).to_numpy();x['high']=(last.loc[x.index,'risk'].to_numpy()>=-5).astype(int)
 else:x=x.join(truth.set_index('event_id'));x['y']=x.true_risk.clip(lower=-15);x['high']=(x.true_risk>=-5).astype(int)
 x['risk_now']=x.risk.clip(lower=-15);x['log_miss']=np.log1p(x.miss_distance);x['log_sigma']=np.log1p((x.t_sigma_r+x.t_sigma_t+x.t_sigma_n+x.c_sigma_r+x.c_sigma_t+x.c_sigma_n).clip(lower=0,upper=1e9))
 return x.dropna(subset=['y'])
u=snap(tr,True);v=snap(te,False);f=['risk_now','log_miss','log_sigma','time_to_tca','t_h_per','t_obs_used','c_obs_used']
reg=make_pipeline(SimpleImputer(strategy='median'),XGBRegressor(n_estimators=150,max_depth=3,learning_rate=.05,n_jobs=4,random_state=42)).fit(u[f],u.y)
prob=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=.1)).fit(u[f],u.high)
pr=prob.predict_proba(v[f])[:,1];threshold=np.quantile(pr,.95);warn=pr>=threshold
official={'train_eligible':len(u),'test_eligible':len(v),'train_high_fraction':float(u.high.mean()),'test_high_fraction':float(v.high.mean()),'test_regression_mae':float(mean_absolute_error(v.y,reg.predict(v[f]))),'test_top5pct_precision':float(precision_score(v.high,warn,zero_division=0)),'test_top5pct_recall':float(recall_score(v.high,warn,zero_division=0)),'warning_fraction':float(warn.mean()),'note':'Official event IDs overlap across files but identify different encounters; the split is kept separate.'}
# Event bootstrap on fixed 72-hour predictions.
w=p[(p.horizon_h==72)&(p.model=='xgboost_warning')].dropna(subset=['warning']);arr=w[['y','warning']].to_numpy(dtype=float).astype(int);boots=[]
for _ in range(1000):
 sample=arr[rng.integers(0,len(arr),len(arr))];boots.append((precision_score(sample[:,0],sample[:,1],zero_division=0),recall_score(sample[:,0],sample[:,1],zero_division=0)))
ci=np.quantile(boots,[.025,.975],axis=0).tolist()
res={'common_horizon_xgboost':common_horizon,'subgroups_72h':sub,'mission_transfer':transfer,'official_48h':official,'warning_bootstrap_72h':{'precision_ci95':[ci[0][0],ci[1][0]],'recall_ci95':[ci[0][1],ci[1][1]],'resample_unit':'event'}}
(O/'robustness.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
