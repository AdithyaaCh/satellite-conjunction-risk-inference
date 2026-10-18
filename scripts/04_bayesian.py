"""Event-level Bayesian regression and a focused conjunction hierarchy (NUTS)."""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd,pymc as pm,arviz as az
from sklearn.preprocessing import StandardScaler
O=Path('outputs/tables');s=pd.read_parquet('outputs/snapshots.parquet');d=pd.read_parquet('outputs/longitudinal.parquet');rng=np.random.default_rng(42)
a=s[(s.horizon_h==72)&s.partition.eq('train')&(s.target_risk_log10>-15)].copy()
a['risk_now']=a.risk.clip(lower=-15);a['log_miss']=np.log1p(a.miss_distance);a['log_sigma']=np.log1p(a.t_sigma_norm+a.c_sigma_norm)
cols=['risk_now','log_miss','log_sigma','n_cdm','t_h_per'];a=a.dropna(subset=cols+['target_risk_capped']).sample(min(900,len(a)),random_state=42)
scaler=StandardScaler().fit(a[cols]);x=scaler.transform(a[cols]);y=(a.target_risk_capped.to_numpy()+10)/5
results={}
for prior_scale in [0.5,2.0]:
 with pm.Model(coords={'term':cols,'event':a.event_id.to_numpy()}) as model:
  beta=pm.Normal('beta',0,prior_scale,dims='term');alpha=pm.Normal('alpha',0,2)
  sigma=pm.HalfNormal('sigma',2);nu=pm.Exponential('nu_minus_2',1/10)+2
  mu=alpha+pm.math.dot(x,beta)
  pm.StudentT('observed',nu=nu,mu=mu,sigma=sigma,observed=y,dims='event')
  trace=pm.sample(300,tune=300,chains=2,cores=2,nuts_sampler='numpyro',target_accept=.9,random_seed=42,progressbar=False,idata_kwargs={'log_likelihood':True})
  ppc=pm.sample_posterior_predictive(trace,var_names=['observed'],random_seed=42,progressbar=False)
 summ=az.summary(trace,var_names=['alpha','beta','sigma'],hdi_prob=.95)
 summ.to_csv(O/f'bayesian_linear_prior_{prior_scale}.csv')
 loo=az.loo(trace,pointwise=True);waic=az.waic(trace,pointwise=True)
 pred=ppc.posterior_predictive['observed'].values.reshape(-1,len(y))
 results[str(prior_scale)]={'n_events':len(a),'estimand':'conditional on final log10 risk > -15','elpd_loo':float(loo.elpd_loo),'p_loo':float(loo.p_loo),'elpd_waic':float(waic.elpd_waic),'p_waic':float(waic.p_waic),'pareto_k_over_0_7':int(np.sum(loo.pareto_k.values>.7)),'rhat_max':float(summ.r_hat.max()),'ess_bulk_min':float(summ.ess_bulk.min()),'divergences':int(trace.sample_stats.diverging.sum()),'ppc_90_coverage_in_sample':float(np.mean((y>=np.quantile(pred,.05,axis=0))&(y<=np.quantile(pred,.95,axis=0))))}
 if prior_scale==2.0:
  trace.to_netcdf('outputs/models/bayesian_linear.nc')
  test=s[(s.horizon_h==72)&s.partition.eq('test')&(s.target_risk_log10>-15)].copy()
  test['risk_now']=test.risk.clip(lower=-15);test['log_miss']=np.log1p(test.miss_distance);test['log_sigma']=np.log1p(test.t_sigma_norm+test.c_sigma_norm)
  test=test.dropna(subset=cols);xt=scaler.transform(test[cols]);post=trace.posterior
  bet=post['beta'].values.reshape(-1,len(cols));al=post['alpha'].values.ravel();sd=post['sigma'].values.ravel();nu=post['nu_minus_2'].values.ravel()+2
  mu=al[:,None]+np.einsum('sk,nk->sn',bet,xt,optimize=False);draws=mu+sd[:,None]*rng.standard_t(nu[:,None],size=mu.shape)
  test_pred=draws*5-10;actual=test.target_risk_log10.to_numpy()
  results['2.0']['conditional_test']={'n':len(test),'mae':float(np.mean(abs(actual-test_pred.mean(axis=0)))),'coverage90':float(np.mean((actual>=np.quantile(test_pred,.05,axis=0))&(actual<=np.quantile(test_pred,.95,axis=0)))),'width90':float(np.mean(np.quantile(test_pred,.95,axis=0)-np.quantile(test_pred,.05,axis=0)))}
  pd.DataFrame({'event_id':a.event_id,'observed':y,'ppc_mean':pred.mean(axis=0),'ppc_low':np.quantile(pred,.05,axis=0),'ppc_high':np.quantile(pred,.95,axis=0)}).to_csv(O/'bayesian_ppc.csv',index=False)
# Longitudinal hierarchy: stable nonfloor event histories with a Gaussian working likelihood.
d=d[d.risk>-15].copy();counts=d.groupby('event_id').size();spread=d.groupby('event_id').risk.agg(lambda x:x.max()-x.min());stable=counts[(counts>=4)&(spread<=5)];ids=stable.sample(100,random_state=42).index
b=d[d.event_id.isin(ids)].copy();b=b.groupby('event_id',sort=False).head(7)
xh=StandardScaler().fit_transform(b[['time_to_tca']]);yh=(b.risk_capped.to_numpy()+10)/5
event_idx=pd.Categorical(b.event_id).codes
with pm.Model() as hm:
 intercept=pm.Normal('intercept',0,2);beta=pm.Normal('beta',0,1,shape=1)
 tau_event=pm.HalfNormal('tau_event',1)
 event_effect=pm.ZeroSumNormal('event_effect',sigma=tau_event,shape=len(np.unique(event_idx)))
 sigma=pm.HalfNormal('sigma',1)
 mu=intercept+pm.math.dot(xh,beta)+event_effect[event_idx]
 pm.Normal('observed',mu=mu,sigma=sigma,observed=yh)
 ht=pm.sample(500,tune=500,chains=4,cores=4,nuts_sampler='numpyro',target_accept=.95,random_seed=43,progressbar=False)
 hp=pm.sample_posterior_predictive(ht,var_names=['observed'],random_seed=43,progressbar=False)
hsum=az.summary(ht,var_names=['intercept','beta','tau_event','sigma'],hdi_prob=.95);hsum.to_csv(O/'hierarchical_coefficients.csv')
hpred=hp.posterior_predictive['observed'].values.reshape(-1,len(yh))
results['hierarchical']={'events':len(ids),'cdms':len(b),'estimand':'conditional on CDM log10 risk > -15 and within-event nonfloor range <= 5','missions':int(b.mission_id.nunique()),'rhat_max':float(az.summary(ht).r_hat.max()),'ess_bulk_min':float(az.summary(ht).ess_bulk.min()),'rhat_max_displayed':float(hsum.r_hat.max()),'ess_bulk_min_displayed':float(hsum.ess_bulk.min()),'divergences':int(ht.sample_stats.diverging.sum()),'ppc_90_coverage_in_sample':float(np.mean((yh>=np.quantile(hpred,.05,axis=0))&(yh<=np.quantile(hpred,.95,axis=0)))),'note':'PPC is in-sample; conditional random effects cannot be used as new-event LOO. Satellite identity is unavailable. Abrupt-revision events are excluded from this focused demonstration after multimodal NUTS diagnostics.'}
ht.to_netcdf('outputs/models/hierarchical.nc')
(O/'bayesian.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
