import argparse
import yaml
import joblib
import pandas as pd
import numpy as np
import os
import itertools
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
import logging

from src.feature_engineering import OutlierRemoval

import warnings
warnings.filterwarnings("ignore")

def generate_all_combinations(steps):
    for r in range(1, len(steps) + 1):
        for combo in itertools.combinations(steps, r):
            yield combo

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    #mkdir for output if not exists
    if not os.path.exists(cfg["gs_output_dir"]):
        os.makedirs(cfg["gs_output_dir"])

    # --- Setup logging ---
    logging.basicConfig(
        filename=os.path.join(cfg["gs_output_dir"], "grid_search.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"starting grid search...")
    
    #load data
    df = pd.read_csv(cfg["train_data_path"])
    if cfg["sample_frac"] is not None:
        df = df.sample(frac=cfg["sample_frac"], random_state=cfg.get("random_state", 42)).reset_index(drop=True)
    X = df[cfg["feature_cols"]]
    y = df[cfg["target_col"]]
    logging.info(f"dataset loaded. X shape : {X.shape}. y shape : {y.shape}.")

    preproc_cfg = cfg["preprocessing"]
    models_cfg = cfg["models_and_params"]
    logging.info(f"preprocessing config : {preproc_cfg}")
    logging.info(f"models config : {models_cfg}")

    # --- Build preprocessing steps ---
    available_steps = []

    if preproc_cfg["polynomial"]["enabled"]:
        available_steps.append(("polynomial", PolynomialFeatures(), preproc_cfg["polynomial"]["degree"]))

    if preproc_cfg["binning"]["enabled"]:
        available_steps.append(("binning", KBinsDiscretizer(encode="ordinal"), preproc_cfg["binning"]["n_bins"]))

    if preproc_cfg["outlier_removal"]["enabled"]:
        available_steps.append(("outlier_removal", OutlierRemoval(), [None])) # no hyperparams

    if preproc_cfg["standardization"]["enabled"]:
        available_steps.append(("standardization", StandardScaler(), [None]))  # no hyperparams

    if preproc_cfg["pca"]["enabled"]:
        available_steps.append(("pca", PCA(), preproc_cfg["pca"]["var_ratio"]))

    preprocessing_combinations = list(generate_all_combinations(available_steps))
    logging.info(f"preprocessing combinations generated. number of combinations : {len(preprocessing_combinations)}.")

    # --- Run GridSearch for each model ---
    results = []
    for model_name, model_cfg in models_cfg.items():
        model_class = eval(model_cfg["model"])  # e.g. "Ridge" -> Ridge
        model = model_class()

        for combo in preprocessing_combinations:
            steps = []
            param_grid = {}

            for step_name, transformer, param_values in combo:
                steps.append((step_name, transformer))

                if step_name == "polynomial":
                    param_grid[f"{step_name}__degree"] = param_values
                elif step_name == "binning":
                    param_grid[f"{step_name}__n_bins"] = param_values
                elif step_name == "pca":
                    param_grid[f"{step_name}__n_components"] = param_values

            # add model
            steps.append(("model", model))
            param_grid.update(model_cfg["params"])

            pipeline = Pipeline(steps=steps)
            grid = GridSearchCV(pipeline, param_grid, cv=cfg['n_splits'], scoring="neg_mean_squared_error", n_jobs=-1)
            try:
                grid.fit(X.copy(), y.copy())
                result = (model_name, [s[0] for s in combo], grid.best_params_, (-grid.best_score_)**0.5)
                results.append(result)
                logging.info(result)

            except Exception as e:
                logging.info(f"skipping grid search for parameter : {param_grid}. reason : {e}")
                continue

    # --- Display results ---
    for res in results:
        model, preproc_steps, best_params, rmse = res
        print(f"Model: {model}, Preprocessing: {preproc_steps}, RMSE: {rmse:.4f}")
        print(f"Best Params: {best_params}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)