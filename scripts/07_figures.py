import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
O=Path('outputs/figures');O.mkdir(exist_ok=True)
m=pd.read_csv('outputs/tables/predictive_metrics.csv');n=pd.read_csv('outputs/tables/sequence_metrics.csv');r=json.load(open('outputs/tables/robustness.json'))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for name,g in pd.concat([m[m.task=='regression'][['horizon_h','model','mae','interval_coverage90']],n[['horizon_h','model','mae','interval_coverage90']]]).groupby('model'):
 if name in ['linear','forest','xgboost','gru','lstm','prob_gru']:
  g=g.sort_values('horizon_h');ax[0].plot(g.horizon_h,g.mae,marker='o',label=name);ax[1].plot(g.horizon_h,g.interval_coverage90,marker='o',label=name)
ax[0].set(xlabel='Hours before TCA',ylabel='MAE on capped log10 risk',title='Held-out accuracy');ax[1].set(xlabel='Hours before TCA',ylabel='90% interval coverage',title='Held-out interval calibration',ylim=(.83,.98));ax[1].axhline(.9,ls='--',color='black',lw=1);ax[0].legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(O/'model_comparison.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(5.4,3.7));g=pd.DataFrame(r['common_horizon_xgboost']).sort_values('horizon_h');ax.plot(g.horizon_h,g.mae,'o-',color='#214e76',lw=2);ax.set(xlabel='Hours before TCA',ylabel='MAE on common 1,251 events',title='Prediction improves as TCA approaches');fig.tight_layout();fig.savefig(O/'paired_horizons.png',dpi=200);plt.close(fig)
w=m[(m.task=='warning')&(m.model.isin(['logistic','xgboost']))];fig,ax=plt.subplots(1,2,figsize=(9,3.5));
for name,g in w.groupby('model'):
 g=g.sort_values('horizon_h');ax[0].plot(g.horizon_h,g.precision,'o-',label=name);ax[1].plot(g.horizon_h,g.recall,'o-',label=name)
for a,title,y in zip(ax,['Warning precision','Warning recall'],['Precision','Recall']):a.set(xlabel='Hours before TCA',ylabel=y,title=title,ylim=(0,1));a.legend()
fig.tight_layout();fig.savefig(O/'warnings.png',dpi=200);plt.close(fig)
p=pd.read_csv('outputs/tables/test_predictions.csv');q=p[(p.horizon_h==72)&(p.model=='xgboost_warning')].sort_values('prediction');bins=np.array_split(q,8);obs=[];pred=[];low=[];high=[]
for b in bins:
 k=b.y.sum();n=len(b);obs.append(k/n);pred.append(b.prediction.mean());low.append(beta.ppf(.025,k+.5,n-k+.5));high.append(beta.ppf(.975,k+.5,n-k+.5))
fig,ax=plt.subplots(figsize=(5,3.7));ax.errorbar(pred,obs,yerr=[np.maximum(0,np.array(obs)-low),np.maximum(0,np.array(high)-obs)],fmt='o',capsize=2);ax.plot([0,max(pred)*1.1],[0,max(pred)*1.1],'--',color='gray');ax.set(xlabel='Mean predicted probability',ylabel='Observed final-risk fraction',title='72-hour warning calibration (8 equal-count bins)');fig.tight_layout();fig.savefig(O/'warning_calibration.png',dpi=200);plt.close(fig)
