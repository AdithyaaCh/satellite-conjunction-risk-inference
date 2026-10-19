# Satellite Conjunction Risk: Statistical Analysis

Satellite collision-risk estimates get revised as new tracking data arrives. I used the [ESA Collision Avoidance Challenge dataset](https://zenodo.org/records/4463683) to ask a practical question: how well can an earlier assessment predict the **last reported risk estimate** for the same encounter?

The [five-page project write-up](reports/technical_report.pdf) explains the setup and results without the detail of every experiment. The outcome here is a later *calculated estimate*, not an observed collision.

## What I found

- On the same 1,251 test encounters, XGBoost's mean absolute error fell from **0.938 at 120 hours** before closest approach to **0.640 at 48 hours**.
- At 72 hours, tree models and a GRU beat the linear regression baseline on final-risk estimation. XGBoost reduced MAE by **0.176** versus linear regression (event-bootstrap 95% interval: 0.144 to 0.205).
- Good average error did not make for clean alerts. The 72-hour XGBoost warning caught **4 of 7** encounters with high final reported risk, alongside **69 false alarms**. These are small counts, so the warning result is fragile.
- Intervals calibrated on held-out encounters reached roughly **90–92% coverage** at a 90% target. A Bayesian model for the nonfloor subset undercovered, despite acceptable sampling diagnostics.

## How I kept the comparison fair

The archive has 199,082 conjunction data messages (CDMs) from 15,321 encounters. For each prediction horizon—120, 96, 72 or 48 hours—I used only CDMs available by that cutoff. All records from an encounter stay in one train, validation, calibration or test partition. The last CDM supplies the target only after the cutoff. [`scripts/08_validate.py`](scripts/08_validate.py) checks these rules against the source data.

The risk field has a large floor at log₁₀ risk = −30. Regression caps final log-risk below at −15; the warning task asks whether the final reported risk reaches 10⁻⁵. The archive was screened before release, high-risk encounters are rare, and spacecraft identities and calendar dates are unavailable. That limits how far these test results can be carried into operations.

## Files

- [`reports/technical_report.pdf`](reports/technical_report.pdf): short write-up. The editable `.tex` file is in the same folder.
- [`src/conjunctions/data.py`](src/conjunctions/data.py): loading, event splits and horizon snapshots.
- [`scripts/`](scripts/): numbered analysis steps.
- [`outputs/tables/`](outputs/tables/), [`outputs/figures/`](outputs/figures/) and [`outputs/models/`](outputs/models/): saved results, plots and model states.
- [`requirements.lock`](requirements.lock): Python packages used for the recorded run.

Raw data, logs and intermediate Parquet files are not tracked in Git.

## Run it

Use Python 3.10 and [Tectonic](https://tectonic-typesetting.github.io/) for the PDF. Download the [ESA archive](https://zenodo.org/records/4463683) and arrange the files like this:

```text
Collision Avoidance Challenge - Dataset/
├── raw_data/raw_data_2015-2019.gz
└── kelvins_competition_data/
    ├── train_data.zip
    ├── test_data.csv
    └── test_data_private.csv
```

From the repository root:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
for step in 01_prepare 02_frequentist 03_cross_validate 03_predict 04_bayesian 05_sequences 06_robustness 07_figures 08_validate 09_report; do
  PYTENSOR_FLAGS='cxx=' .venv/bin/python "scripts/$step.py"
done
```

On macOS, XGBoost may need `brew install libomp`. The PyTensor setting handles the Apple command-line SDK used for this run. Bayesian sampling and sequence training take the longest.

Dataset details: [ESA column descriptions](https://kelvins.esa.int/collision-avoidance-challenge/data/) and the [original challenge overview](https://arxiv.org/pdf/2008.03069.pdf).
