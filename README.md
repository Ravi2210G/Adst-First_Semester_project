# AI Dataset Health Inspector

A desktop tool for diagnosing common dataset quality issues before machine learning. It imports CSV and Excel files, calculates dataset and ML-readiness scores, explains statistical findings, and exports a cleaned copy without changing the source file.

## Features

- CSV, `.xlsx`, and `.xls` import
- Overview metrics for dataset size, missing cells, and duplicate rows
- Per-column type, missingness, and unique-value profile
- Detection of missing values, exact duplicates, constant columns, potentially mixed numeric/text values, IQR outliers, high numeric correlations, and strongly skewed numeric features
- Optional classification-target check for class imbalance
- Dataset health and ML-readiness scores
- Conservative cleaning export with optional numeric-median/categorical-mode imputation and exact duplicate removal
- Data preview of up to 200 rows

The inspector uses statistical and rule-based analysis, not an LLM. Its findings are signals for review, not domain-specific guarantees. Outliers are reported rather than automatically removed, and a classification target is only assessed when selected.

## Run

Python 3.10 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

## Test

```powershell
python -m unittest discover -s tests -v
```
