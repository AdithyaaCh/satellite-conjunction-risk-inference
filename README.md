# Satellite Conjunction Risk Inference

Statistical analysis of how satellite collision-risk **estimates** change as a close approach nears. Using the [ESA Collision Avoidance Challenge dataset](https://zenodo.org/records/4463683), this project compares frequentist and Bayesian inference with tree and sequence models at 120, 96, 72, and 48 hours before time of closest approach (TCA).

**Read the [14-page research report](reports/technical_report.pdf)** for the methods, diagnostics, limitations, and full results. The outcome is the **last reported risk estimate for an encounter**. The data do not contain observed collisions, so these models do not estimate physical collision frequency.

## Results at a glance

| Finding | Held-out result |
|---|---:|
| XGBoost MAE on the same 1,251 events | 0.938 at 120 h; 0.640 at 48 h |
| XGBoost versus linear regression at 72 h | 0.176 lower MAE; event-bootstrap 95% CI 0.144–0.205 |
| 72 h warning for final reported risk ≥ 10⁻⁵ | 4 of 7 events detected; 69 false alarms; 5.5% precision |

The rare high-risk category makes warning estimates unstable. Marginal 90% conformal intervals covered roughly 90–92% of held-out outcomes; a focused Bayesian model's intervals undercovered. See the report before interpreting any model as an operational warning system.

## Repository layout

| Path | Contents |
|---|---|
| [`reports/technical_report.pdf`](reports/technical_report.pdf) | Finished paper; editable LaTeX source is alongside it |
| [`src/conjunctions/data.py`](src/conjunctions/data.py) | Event partitions and leakage-aware horizon snapshots |
| [`scripts/`](scripts/) | Numbered preparation, analysis, validation, and report steps |
| [`outputs/tables/`](outputs/tables/) | Result summaries, coefficients, and prediction-level evidence |
| [`outputs/figures/`](outputs/figures/) | Figures used in the paper |
| [`outputs/models/`](outputs/models/) | Saved neural-network states and Bayesian posterior draws |
| [`requirements.lock`](requirements.lock) | Exact Python package versions used for this run |

Raw ESA data, the Python environment, logs, and intermediate Parquet files are excluded from Git. The data remain under their original ESA terms at the linked source.

## Reproduce

Use Python 3.10 and [Tectonic](https://tectonic-typesetting.github.io/) for the PDF. On macOS, XGBoost may also require `brew install libomp`.

1. Download the archive from [Zenodo](https://zenodo.org/records/4463683). Place its files so these paths exist:

   ```text
   Collision Avoidance Challenge - Dataset/
   ├── raw_data/raw_data_2015-2019.gz
   └── kelvins_competition_data/
       ├── train_data.zip
       ├── test_data.csv
       └── test_data_private.csv
   ```

2. From the repository root, install dependencies and run the numbered scripts:

   ```bash
   python3.10 -m venv .venv
   .venv/bin/python -m pip install -r requirements.lock
   for step in 01_prepare 02_frequentist 03_cross_validate 03_predict 04_bayesian 05_sequences 06_robustness 07_figures 08_validate 09_report; do
     PYTENSOR_FLAGS='cxx=' .venv/bin/python "scripts/$step.py"
   done
   ```

   The PyTensor setting handles the Apple command-line SDK used in the recorded run. Bayesian sampling and sequence training take longer than the other steps. `08_validate.py` checks event separation, time cutoffs, and target construction before `09_report.py` rebuilds the PDF.

## Study design

The 199,082 conjunction data messages represent 15,321 anonymized encounters. At each horizon, predictors use only messages available by that cutoff; the target is the event's later final message, required to be within 24 hours of TCA. Entire events share one train, validation, calibration, or test partition. Regression uses final log₁₀ risk capped below at −15, and the warning task labels final reported risk ≥ 10⁻⁵. [The paper](reports/technical_report.pdf) gives the cohort sizes and all model specifications.

The archive is selected through ESA screening and its competition test set is enriched for higher risk. Exact satellite identities and calendar dates are unavailable, so satellite-disjoint and chronological tests cannot be performed. Inference applies to the eligible archive encounters and to **future reported risk estimates**, not to observed impacts.

## Sources

- [ESA dataset, Zenodo DOI 10.5281/zenodo.4463683](https://doi.org/10.5281/zenodo.4463683)
- [ESA data and column descriptions](https://kelvins.esa.int/collision-avoidance-challenge/data/)
- [Uriot et al. (2020), challenge design and results](https://arxiv.org/pdf/2008.03069.pdf)
