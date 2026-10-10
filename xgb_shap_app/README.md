# XGBoost + TreeSHAP

A Streamlit GUI on top of an XGBoost engine with exact TreeSHAP explanations.
It takes `.csv` or `.xlsx` files as input.

## Run

```bash
./run.sh
```

Then open http://localhost:8501. To set it up from scratch, run
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Files

- `engine.py`: loading, preprocessing, training, metrics and TreeSHAP. It has no GUI code, so you can use it from scripts.
- `app.py`: the Streamlit GUI.
- `make_samples.py`: writes demo data to `sample_data/`.

## Features

- Detects the task type (regression, binary or multiclass). Uses XGBoost's native categorical support, turns date columns into numbers, and lets XGBoost handle missing values.
- Splits the data into train, validation and test sets. Uses early stopping, has optional class balancing and can run on a GPU.
- Shows metrics, a learning curve, a confusion matrix, an ROC curve, predicted vs actual, and residuals.
- TreeSHAP views: global importance, beeswarm, dependence plots with interaction colouring, waterfall and force plots for a single row, and an interaction-value heatmap.
- Scores new files and lets you download the predictions and per-row SHAP values as CSV or Excel.
- Lets you load a saved model in the Predict tab and score new data, without training data. Models exported by this app restore exactly. Other XGBoost `.json`/`.ubj` models also load, and the app warns about anything it can't recover.
- Exports the model as JSON and a SHAP report as Excel.

SHAP values come from XGBoost's own TreeSHAP (`pred_contribs` and
`pred_interactions`). They are exact, and they work with categorical
features. They are in margin space: target units for regression and
log-odds for classification.

## Use the engine from a script

```python
import engine as eng
df = eng.load_table("data.xlsx", "data.xlsx", sheet="Sheet1")
r = eng.train(df, target="y", features=[c for c in df if c != "y"], task=eng.infer_task(df["y"]))
print(r.metrics)
eng.shap_frame(r, df).to_csv("scored_with_shap.csv", index=False)
```
