import json,sys,warnings
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np,pandas as pd
np.random.seed(42)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan,linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
O=Path('outputs/tables');O.mkdir(parents=True,exist_ok=True)
s=pd.read_parquet('outputs/snapshots.parquet'); l=pd.read_parquet('outputs/longitudinal.parquet')
a=s[(s.horizon_h==72)&s.partition.eq('train')].copy()
a['log_miss']=np.log1p(a.miss_distance.clip(lower=0));a['log_sigma']=np.log1p(a.t_sigma_norm+a.c_sigma_norm)
a['risk_now']=a.risk.clip(lower=-15);a['risk_now_c']=a.risk_now-a.risk_now.mean();a['log_sigma_c']=a.log_sigma-a.log_sigma.mean();a['log_obs']=np.log1p(a.t_obs_used.clip(lower=0));a['log_rcs']=np.log1p(a.c_rcs_estimate.clip(lower=0))
# Linear MLE, heteroscedasticity-robust inference, and nested likelihood tests.
f='target_risk_capped ~ risk_now_c + log_miss + log_sigma_c + log_obs + n_cdm + C(regime) + risk_now_c:log_sigma_c'
fit=smf.ols(f,data=a).fit();rob=fit.get_robustcov_results(cov_type='HC3')
coef=pd.DataFrame({'term':fit.params.index,'estimate':rob.params,'se_hc3':rob.bse,'p_hc3':rob.pvalues,'ci_low':rob.conf_int()[:,0],'ci_high':rob.conf_int()[:,1]})
coef.to_csv(O/'ols_coefficients.csv',index=False)
null=smf.ols('target_risk_capped ~ risk_now_c + log_miss + log_sigma_c + log_obs + n_cdm + C(regime)',data=a).fit()
lr=2*(fit.llf-null.llf);lrt_p=stats.chi2.sf(lr,fit.df_model-null.df_model)
# GLM on threshold -5; the output is a final risk category, not a collision outcome.
glm=smf.glm('high_risk_1e5 ~ risk_now + log_miss + log_sigma + log_obs + n_cdm + C(regime)',data=a,family=sm.families.Binomial()).fit(cov_type='HC3')
gcoef=pd.DataFrame({'term':glm.params.index,'log_odds':glm.params,'odds_ratio':np.exp(glm.params),'p_hc3':glm.pvalues,'ci_low_or':np.exp(glm.conf_int()[0]),'ci_high_or':np.exp(glm.conf_int()[1])})
gcoef.to_csv(O/'glm_coefficients.csv',index=False)
gnull=smf.glm('high_risk_1e5 ~ risk_now + log_miss + log_sigma + log_obs + n_cdm',data=a,family=sm.families.Binomial()).fit()
glr=2*(glm.llf-gnull.llf);glrt_p=stats.chi2.sf(glr,glm.df_model-gnull.df_model)
# Confounding: compare current-risk effect with and without measured geometry/uncertainty.
unadj=smf.ols('target_risk_capped ~ risk_now_c',data=a).fit(cov_type='HC3')
# Explicit missingness model; object type and altitude may explain unavailable radar cross section.
a['rcs_missing']=a.c_rcs_estimate.isna().astype(int)
miss=smf.glm('rcs_missing ~ risk_now + log_miss + C(c_object_type) + C(regime)',data=a,family=sm.families.Binomial()).fit()
# Multiple imputation under conditional MAR for the RCS sensitivity estimand.
mi_result={'status':'not run'}
try:
    from statsmodels.imputation.mice import MICEData
    cols=['target_risk_capped','risk_now','log_miss','log_sigma','log_rcs']
    md=MICEData(a[cols].replace([np.inf,-np.inf],np.nan).sample(min(2000,len(a)),random_state=42))
    estimates=[];variances=[]
    for j in range(8):
        md.update_all(2)
        if j>=3:
            fmi=smf.ols('target_risk_capped ~ risk_now + log_miss + log_sigma + log_rcs',data=md.data).fit()
            estimates.append(fmi.params['log_rcs']);variances.append(fmi.bse['log_rcs']**2)
    q=np.mean(estimates);u=np.mean(variances);b=np.var(estimates,ddof=1);se=np.sqrt(u+(1+1/5)*b)
    mi_result={'status':'ok','n_imputations':5,'rcs_coef':float(q),'se_rubin':float(se),'ci_low':float(q-1.96*se),'ci_high':float(q+1.96*se),'assumption':'conditional MAR under included variables; sensitivity only'}
except Exception as e: mi_result={'status':'failed','reason':repr(e)}
cc=smf.ols('target_risk_capped ~ risk_now + log_miss + log_sigma + log_rcs',data=a).fit()
# Longitudinal mixed model on a fixed event sample, random intercept and bounded outcome diagnostic caveat.
counts=l.groupby('event_id').size();ids=counts[counts>=3].sample(min(220,(counts>=3).sum()),random_state=42).index
m=l[l.event_id.isin(ids)].copy();m['tca_c']=m.time_to_tca-m.time_to_tca.mean();m['log_miss_c']=m.log_miss-m.log_miss.mean();m['log_sigma_c']=m.log_sigma-m.log_sigma.mean()
mixed_result={'status':'not run'}
try:
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always')
        mf=smf.mixedlm('risk_capped ~ tca_c + log_miss_c + log_sigma_c',data=m,groups=m.event_id).fit(method='lbfgs',reml=False,maxiter=200)
    rand=float(mf.cov_re.iloc[0,0]);resid=float(mf.scale)
    mixed_result={'status':'ok','events':len(ids),'cdms':len(m),'loglik':float(mf.llf),'aic':float(mf.aic),'bic':float(mf.bic),'icc':rand/(rand+resid),'random_intercept_variance':rand,'residual_variance':resid,'converged':bool(mf.converged),'warnings':[str(x.message) for x in ww][:4]}
    pd.DataFrame({'term':mf.params.index,'estimate':mf.params.to_numpy(),'se':mf.bse.to_numpy(),'p':mf.pvalues.to_numpy()}).to_csv(O/'mixed_coefficients.csv',index=False)
except Exception as e:mixed_result={'status':'failed','reason':repr(e)}
# Diagnostics are descriptive for the capped, non-normal target.
bp=het_breuschpagan(fit.resid,fit.model.exog)
try:reset=linear_reset(fit,power=2,use_f=True)
except Exception:reset=None
vif={name:float(variance_inflation_factor(fit.model.exog,i)) for i,name in enumerate(fit.model.exog_names) if name!='Intercept'}
influence=fit.get_influence().cooks_distance[0]
result={'n':len(a),'positive_train':int(a.high_risk_1e5.sum()),'ols_r2':float(fit.rsquared),'ols_aic':float(fit.aic),'ols_bic':float(fit.bic),'interaction_lrt_stat':float(lr),'interaction_lrt_p':float(lrt_p),'glm_aic':float(glm.aic),'glm_bic_llf':float(glm.bic_llf),'regime_lrt_stat':float(glr),'regime_lrt_p':float(glrt_p),'risk_now_unadjusted':float(unadj.params['risk_now_c']),'risk_now_adjusted':float(fit.params['risk_now_c']),'missing_rcs_fraction':float(a.rcs_missing.mean()),'missingness_model_lr_p':float(stats.chi2.sf(miss.null_deviance-miss.deviance,miss.df_model)),'complete_case_rcs_coef':float(cc.params['log_rcs']),'multiple_imputation':mi_result,'mixed_effects':mixed_result,'breusch_pagan_p':float(bp[1]),'jarque_bera_p':float(stats.jarque_bera(fit.resid).pvalue),'reset_p':float(reset.pvalue) if reset is not None else None,'max_vif':max(vif.values()),'vif':vif,'max_cooks_distance':float(np.nanmax(influence)),'influential_fraction_4_over_n':float((influence>4/len(a)).mean())}
(O/'frequentist.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
