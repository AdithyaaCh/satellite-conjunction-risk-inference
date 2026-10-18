"""Grouped training-only CV for compact 72-hour hyperparameter sensitivity."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
s=pd.read_parquet('outputs/snapshots.parquet');a=s[(s.horizon_h==72)&s.partition.eq('train')].copy()
a['risk']=a.risk.clip(lower=-15);a['delta_risk']=a.delta_risk.clip(-15,15)
for c in ['miss_distance','t_sigma_norm','c_sigma_norm','t_obs_used','c_obs_used']:a[c]=np.log1p(a[c].clip(lower=0))
features=['risk','delta_risk','miss_distance','delta_miss_distance','relative_speed','t_sigma_norm','c_sigma_norm','delta_t_sigma_r','delta_c_sigma_r','t_h_per','t_j2k_inc','t_obs_used','c_obs_used','n_cdm','history_days','mission_id','snapshot_tca_h']
x=a[features].replace([np.inf,-np.inf],np.nan);y=a.target_risk_capped;g=a.event_id
cv=GroupKFold(n_splits=3);rows=[]
for model,grid in [('ridge',[1,10,100]),('forest',[5,8]),('xgboost',[2,3,4])]:
 for value in grid:
  scores=[]
  for i,j in cv.split(x,y,g):
   if model=='ridge':m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=value))
   elif model=='forest':m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),RandomForestRegressor(n_estimators=120,min_samples_leaf=value,max_features=.8,n_jobs=4,random_state=42))
   else:m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),XGBRegressor(n_estimators=180,max_depth=value,learning_rate=.05,subsample=.8,colsample_bytree=.8,n_jobs=4,random_state=42))
   m.fit(x.iloc[i],y.iloc[i]);scores.append(mean_absolute_error(y.iloc[j],m.predict(x.iloc[j])))
  rows.append({'model':model,'parameter':int(value),'cv_mae_mean':float(np.mean(scores)),'cv_mae_sd':float(np.std(scores,ddof=1))})
pd.DataFrame(rows).to_csv('outputs/tables/grouped_cv.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
