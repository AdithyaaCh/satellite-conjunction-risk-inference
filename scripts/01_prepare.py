import json, sys
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np, pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from conjunctions.data import load_raw,event_partition,make_snapshots,make_longitudinal
OUT=Path('outputs'); (OUT/'tables').mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
d=load_raw(); p=event_partition(d); s=make_snapshots(d,p); l=make_longitudinal(d)
p.to_csv(OUT/'tables/partitions.csv',index=False)
s.to_parquet(OUT/'snapshots.parquet',index=False)
l.to_parquet(OUT/'longitudinal.parquet',index=False)
last=d.groupby('event_id',sort=False).tail(1)
summary={
 'cdms':len(d),'events':d.event_id.nunique(),'columns':d.shape[1],
 'events_final_within_24h':int((last.time_to_tca<=1).sum()),
 'high_risk_1e5_all':int((last.risk>=-5).sum()),
 'high_risk_1e4_all':int((last.risk>=-4).sum()),
 'floor_final_fraction':float((last.risk<=-30).mean()),
 'mission_count':int(d.mission_id.nunique()),
 'per_horizon':{str(h):{'events':len(a),'high_risk_1e5':int(a.high_risk_1e5.sum()),'train':int((a.partition=='train').sum()),'test':int((a.partition=='test').sum())} for h,a in s.groupby('horizon_h')},
 'top_missing':d.isna().mean().sort_values(ascending=False).head(15).round(4).to_dict(),
 'risk_missing':float(d.risk.isna().mean()),
 'duplicate_event_time':int(d.duplicated(['event_id','time_to_tca']).sum()),
 'note':'Dates and satellite identities are anonymised. Event partition is disjoint; mission transfer is a separate sensitivity analysis.'
}
(OUT/'tables/audit.json').write_text(json.dumps(summary,indent=2))
d.isna().mean().sort_values(ascending=False).rename('missing_fraction').to_csv(OUT/'tables/missingness.csv')
# Cohort and sequential plots
fig,ax=plt.subplots(1,2,figsize=(10,3.5))
ax[0].hist(last.risk,bins=np.linspace(-30,0,61),color='#336699');ax[0].set(xlabel='Final log10 risk estimate',ylabel='Events',title='Final risk is strongly floor censored')
ax[1].hist(d.groupby('event_id').size(),bins=np.arange(1,25)-.5,color='#bc6c25');ax[1].set(xlabel='CDMs per event',ylabel='Events',title='Unequal observation histories')
fig.tight_layout();fig.savefig(OUT/'figures/distributions.png',dpi=180);plt.close(fig)
# First/last paired comparison, only near-TCA final
z=d.loc[d.event_id.isin(last.loc[last.time_to_tca<=1,'event_id'])]
g=z.groupby('event_id',sort=False); first=g.head(1).set_index('event_id'); final=g.tail(1).set_index('event_id')
paired=pd.DataFrame({'initial_risk':first.risk,'final_risk':final.risk,'initial_miss':first.miss_distance,'final_miss':final.miss_distance,'initial_sigma':l.set_index(['event_id','time_to_tca']).loc[pd.MultiIndex.from_frame(first.reset_index()[['event_id','time_to_tca']]),'sigma_norm'].to_numpy(),'final_sigma':l.set_index(['event_id','time_to_tca']).loc[pd.MultiIndex.from_frame(final.reset_index()[['event_id','time_to_tca']]),'sigma_norm'].to_numpy()})
paired.to_csv(OUT/'tables/paired_evolution.csv')
fig,ax=plt.subplots(1,2,figsize=(10,3.5));rng=np.random.default_rng(42);sample=paired.sample(min(2000,len(paired)),random_state=42)
ax[0].scatter(sample.initial_risk.clip(-15),sample.final_risk.clip(-15),s=5,alpha=.22);ax[0].plot([-15,0],[-15,0],'k--',lw=1);ax[0].set(xlabel='First log10 risk (capped -15)',ylabel='Final log10 risk (capped -15)',title='Risk revisions are heterogeneous')
ax[1].hist(np.log10((paired.final_sigma+1)/(paired.initial_sigma+1)),bins=60,color='#4f7d63');ax[1].set(xlabel='log10 final/initial sigma norm',ylabel='Events',title='Position uncertainty contracts')
fig.tight_layout();fig.savefig(OUT/'figures/evolution.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))
