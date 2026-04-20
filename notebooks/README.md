# Notebooks

- `01_eda.ipynb` — EDA on the base + depth_0/1/2 tables. Check `WEEK_NUM` distribution, target rate drift, missingness.
- `02_baseline_local.ipynb` — Local LightGBM baseline using `src/features/build_features.py` + `src/training/train_lightgbm.py`. Validates the training contract before moving to SageMaker.
- `03_sagemaker_pipeline.ipynb` — Run `src/pipelines/training_pipeline.py` against a real role and inspect artifacts.
