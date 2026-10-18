"""Leakage-aware event snapshots from ESA's 2015-2019 CDM archive."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path('Collision Avoidance Challenge - Dataset/raw_data/raw_data_2015-2019.gz')
HORIZONS = (120, 96, 72, 48)
BASE = ['risk','miss_distance','relative_speed','t_sigma_r','t_sigma_t','t_sigma_n',
        'c_sigma_r','c_sigma_t','c_sigma_n','t_h_per','t_h_apo','t_j2k_inc',
        'c_h_per','c_h_apo','t_obs_used','c_obs_used','c_rcs_estimate','max_risk_estimate']
STATIC = ['mission_id','c_object_type']


def load_raw(path=RAW):
    df = pd.read_csv(path, compression='gzip')
    assert df.event_id.notna().all() and df.time_to_tca.notna().all()
    df = df.sort_values(['event_id','time_to_tca'], ascending=[True,False],kind='stable').reset_index(drop=True)
    assert not df.duplicated(['event_id','time_to_tca']).any(), 'Duplicate event/time requires review'
    return df


def event_partition(df, seed=42):
    """Stratified event partition, fixed across horizons. Label only stratifies partitions."""
    last = df.groupby('event_id',sort=False).tail(1)
    ids = last.event_id.to_numpy()
    labels = (last.risk.to_numpy() >= -5).astype(int)
    train, remain = train_test_split(ids,test_size=.4,stratify=labels,random_state=seed)
    rem_labels = last.set_index('event_id').loc[remain].risk.ge(-5).astype(int).to_numpy()
    valid, rest = train_test_split(remain,test_size=.625,stratify=rem_labels,random_state=seed+1)
    rest_labels = last.set_index('event_id').loc[rest].risk.ge(-5).astype(int).to_numpy()
    calib, test = train_test_split(rest,test_size=.6,stratify=rest_labels,random_state=seed+2)
    out = pd.DataFrame({'event_id':ids})
    out['partition']='train'
    for name,arr in [('validation',valid),('calibration',calib),('test',test)]:
        out.loc[out.event_id.isin(arr),'partition']=name
    assert out.event_id.is_unique
    return out


def make_snapshots(df, partitions, horizons=HORIZONS):
    """At h hours select the latest CDM with time_to_tca >= h/24."""
    last = df.groupby('event_id',sort=False).tail(1).set_index('event_id')
    first = df.groupby('event_id',sort=False).head(1).set_index('event_id')
    rows=[]
    for h in horizons:
        past = df.loc[df.time_to_tca >= h/24]
        snap = past.groupby('event_id',sort=False).tail(1).copy().set_index('event_id')
        # The outcome must be observed strictly after the horizon; near-TCA labels only.
        snap = snap.loc[(last.loc[snap.index,'time_to_tca'].to_numpy() < h/24) &
                        (last.loc[snap.index,'time_to_tca'].to_numpy() <= 1)]
        x = pd.DataFrame(index=snap.index)
        x['horizon_h']=h
        x['snapshot_tca_h']=24*snap.time_to_tca
        x['n_cdm']=past.groupby('event_id').size().loc[snap.index].to_numpy()
        x['history_days']=(first.loc[snap.index,'time_to_tca'].to_numpy()-snap.time_to_tca.to_numpy())
        for col in BASE:
            x[col]=snap[col].to_numpy()
            x[f'delta_{col}']=snap[col].to_numpy()-first.loc[snap.index,col].to_numpy()
        for col in STATIC:
            x[col]=snap[col].to_numpy()
        for prefix in ['t','c']:
            sig = np.column_stack([snap[f'{prefix}_sigma_{d}'] for d in 'rtn'])
            x[f'{prefix}_sigma_norm']=np.sqrt(np.square(sig).sum(axis=1))
        x['target_risk_log10']=last.loc[snap.index,'risk'].to_numpy()
        x['target_risk_capped']=np.maximum(x.target_risk_log10,-15)
        x['high_risk_1e5']=(x.target_risk_log10>=-5).astype(int)
        x['warning_1e6']=(x.target_risk_log10>=-6).astype(int)
        x['final_tca_h']=last.loc[snap.index,'time_to_tca'].to_numpy()*24
        x['regime']=pd.cut(x.t_h_per,[-np.inf,550,700,np.inf],labels=['low LEO','middle LEO','upper LEO']).astype(str)
        rows.append(x.reset_index())
    out=pd.concat(rows,ignore_index=True).merge(partitions,on='event_id',validate='many_to_one')
    assert out.groupby('event_id').partition.nunique().max()==1
    assert (out.snapshot_tca_h>=out.horizon_h).all()
    assert (out.final_tca_h<out.horizon_h).all()
    assert not out.duplicated(['event_id','horizon_h']).any()
    return out


def make_longitudinal(df):
    d=df[['event_id','time_to_tca','mission_id','risk','miss_distance','t_sigma_r','t_sigma_t','t_sigma_n','c_sigma_r','c_sigma_t','c_sigma_n','t_h_per','relative_speed','t_obs_used','c_obs_used']].copy()
    d['risk_capped']=d.risk.clip(lower=-15)
    d['sigma_norm']=np.sqrt(sum(np.square(d[f'{p}_sigma_{a}']) for p in 'tc' for a in 'rtn'))
    d['log_sigma']=np.log1p(d.sigma_norm)
    d['log_miss']=np.log1p(d.miss_distance.clip(lower=0))
    d['altitude_band']=pd.cut(d.t_h_per,[-np.inf,550,700,np.inf],labels=['low LEO','middle LEO','upper LEO']).astype(str)
    return d
