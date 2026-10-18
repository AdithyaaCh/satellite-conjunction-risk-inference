"""Four fixed-cutoff comparisons; all tuning and calibration precede test access."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,brier_score_loss,average_precision_score,precision_score,recall_score,confusion_matrix,log_loss
from xgboost import XGBRegressor,XGBClassifier
from scipy.special import expit
from scipy.optimize import minimize_scalar
s=pd.read_parquet('outputs/snapshots.parquet');O=Path('outputs/tables')
features=['risk','delta_risk','miss_distance','delta_miss_distance','relative_speed','t_sigma_norm','c_sigma_norm','delta_t_sigma_r','delta_c_sigma_r','t_h_per','t_j2k_inc','t_obs_used','c_obs_used','n_cdm','history_days','mission_id','snapshot_tca_h']
s['risk']=s.risk.clip(lower=-15);s['delta_risk']=s.delta_risk.clip(-15,15)
for c in ['miss_distance','t_sigma_norm','c_sigma_norm','t_obs_used','c_obs_used']:
 s[c]=np.log1p(s[c].clip(lower=0))
# Validation chooses a compact tree size. Calibration determines conformal width and warning threshold.
rows=[];preds=[];rng=np.random.default_rng(42)
for h,a in s.groupby('horizon_h'):
 tr=a[a.partition=='train'];v=a[a.partition=='validation'];ca=a[a.partition=='calibration'];te=a[a.partition=='test'];
 if len(te)==0:continue
 def X(d):return d[features].replace([np.inf,-np.inf],np.nan)
 yt=tr.target_risk_capped;yv=v.target_risk_capped;yc=ca.target_risk_capped;ye=te.target_risk_capped
 reg={
  'linear':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=10)),
  'forest':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestRegressor(n_estimators=120,min_samples_leaf=8,max_features=.8,n_jobs=-1,random_state=42)),
  'xgboost':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),XGBRegressor(n_estimators=180,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,n_jobs=4,random_state=42))}
 for name,model in reg.items():
  model.fit(X(tr),yt);pv=model.predict(X(v))
  # The validation split is used for diagnostic selection; fixed configurations remain comparable.
  pc=model.predict(X(ca));pe=model.predict(X(te));q=np.quantile(abs(yc-pc),np.ceil((len(ca)+1)*.9)/len(ca),method='higher')
  cover=np.mean((ye>=pe-q)&(ye<=pe+q))
  rows.append({'horizon_h':h,'task':'regression','model':name,'n_test':len(te),'validation_mae':mean_absolute_error(yv,pv),'mae':mean_absolute_error(ye,pe),'rmse':np.sqrt(mean_squared_error(ye,pe)),'interval_coverage90':cover,'interval_width90':2*q})
  preds.append(pd.DataFrame({'event_id':te.event_id,'horizon_h':h,'model':name,'y':ye,'prediction':pe,'low':pe-q,'high':pe+q}))
  if h==72 and name=='xgboost':
   import xgboost as xgb
   imp=model[0].transform(X(te.iloc[:150]));sv=model[-1].get_booster().predict(xgb.DMatrix(imp),pred_contribs=True)
   pd.DataFrame({'feature':features,'mean_abs_shap':np.abs(sv[:,:len(features)]).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False).to_csv(O/'shap_72h.csv',index=False)
 ytr=tr.high_risk_1e5; yca=ca.high_risk_1e5;yte=te.high_risk_1e5
 cls={
  'logistic':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',C=.1)),
  'forest':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestClassifier(n_estimators=150,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=42)),
  'xgboost':make_pipeline(SimpleImputer(strategy='median',add_indicator=True),XGBClassifier(n_estimators=180,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8,scale_pos_weight=max(1,(len(tr)-ytr.sum())/max(1,ytr.sum())),n_jobs=4,random_state=42))}
 for name,model in cls.items():
  model.fit(X(tr),ytr);pca=np.clip(model.predict_proba(X(ca))[:,1],1e-5,1-1e-5);pte=np.clip(model.predict_proba(X(te))[:,1],1e-5,1-1e-5)
  # Intercept-only probability recalibration on the held-out calibration partition.
  from scipy.special import logit
  z=logit(pca);offset=minimize_scalar(lambda b:log_loss(yca,expit(z+b),labels=[0,1]),bounds=(-8,8),method='bounded').x
  pte=expit(logit(pte)+offset);pca=expit(z+offset)
  thresholds=np.quantile(pca,np.linspace(.5,.995,60));valid=[t for t in thresholds if np.mean((pca>=t)&(yca==0))/max(np.mean(yca==0),1e-9)<=.05]
  th=float(min(valid,key=lambda t: -recall_score(yca,pca>=t,zero_division=0))) if valid else 1.0
  warn=pte>=th;tn,fp,fn,tp=confusion_matrix(yte,warn,labels=[0,1]).ravel()
  rows.append({'horizon_h':h,'task':'warning','model':name,'n_test':len(te),'positives_test':int(yte.sum()),'brier':brier_score_loss(yte,pte),'log_loss':log_loss(yte,pte,labels=[0,1]),'average_precision':average_precision_score(yte,pte),'threshold':th,'precision':precision_score(yte,warn,zero_division=0),'recall':recall_score(yte,warn,zero_division=0),'false_positive_rate':fp/(fp+tn),'false_alarms':int(fp),'missed_warnings':int(fn)})
  preds.append(pd.DataFrame({'event_id':te.event_id,'horizon_h':h,'model':name+'_warning','y':yte,'prediction':pte,'warning':warn}))
res=pd.DataFrame(rows);res.to_csv(O/'predictive_metrics.csv',index=False);pd.concat(preds).to_csv(O/'test_predictions.csv',index=False)
# Event bootstrap for paired 72-h MAE difference, XGBoost versus linear.
pd0=pd.concat(preds);q=pd0[(pd0.horizon_h==72)&pd0.model.isin(['linear','xgboost'])].pivot(index='event_id',columns='model',values='prediction');y=te.set_index('event_id').target_risk_capped if False else s[(s.horizon_h==72)&(s.partition=='test')].set_index('event_id').target_risk_capped
q=q.join(y.rename('y')).dropna();diff=(q.y-q.xgboost).abs()-(q.y-q.linear).abs();boots=[diff.iloc[rng.integers(0,len(diff),len(diff))].mean() for _ in range(1000)]
(O/'bootstrap.json').write_text(json.dumps({'72h_xgb_minus_linear_mae':float(diff.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'unit':'event'}))
print(res.to_string(index=False));print((O/'bootstrap.json').read_text())
