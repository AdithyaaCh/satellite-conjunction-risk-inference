"""Independent invariants for horizon and partition leakage."""
import sys
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from conjunctions.data import load_raw
s=pd.read_parquet('outputs/snapshots.parquet');d=load_raw();p=pd.read_csv('outputs/tables/partitions.csv')
assert p.event_id.is_unique and set(p.partition)=={'train','validation','calibration','test'}
assert s.groupby('event_id').partition.nunique().max()==1
assert not s.duplicated(['event_id','horizon_h']).any()
assert (s.snapshot_tca_h>=s.horizon_h).all() and (s.final_tca_h<s.horizon_h).all()
assert (s.final_tca_h<=24).all()
for row in s.sample(200,random_state=42).itertuples():
 eligible=d[(d.event_id==row.event_id)&(d.time_to_tca>=row.horizon_h/24)]
 assert np.isclose(eligible.iloc[-1].time_to_tca*24,row.snapshot_tca_h)
 assert np.isclose(eligible.iloc[-1].risk,row.risk)
 assert np.isclose(d[d.event_id==row.event_id].iloc[-1].risk,row.target_risk_log10)
print('Leakage and horizon invariants passed:',len(s),'snapshots')
