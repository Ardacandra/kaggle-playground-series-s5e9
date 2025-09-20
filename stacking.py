import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

import os
import joblib
import argparse
import yaml
import logging

import warnings
warnings.filterwarnings("ignore")

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    #mkdir for output if not exists
    if not os.path.exists(cfg["stacking_output_path"]):
        os.makedirs(cfg["stacking_output_path"])

    # --- Setup logging ---
    logging.basicConfig(
        filename=os.path.join(cfg["stacking_output_path"], f"{cfg['stacking_model_name']}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"starting stacking...")

    #load data
    df_train = pd.read_csv(cfg["train_data_path"])
    df_test = pd.read_csv(cfg["test_data_path"])

    target_col = cfg["target_col"]
    feature_cols = cfg["feature_cols"]

    X_train = df_train[feature_cols].copy().reset_index(drop=True)
    y_train = df_train[target_col].copy().reset_index(drop=True)
    X_test = df_test[feature_cols].copy().reset_index(drop=True)
    
    logging.info(f"finished loading train/test data. X_train shape : {X_train.shape}, y_train shape {y_train.shape}, X_test shape : {X_test.shape}.")

    #extract individual models
    models_dict = {}

    for m in cfg["stacking_models_list"]:
        blueprint = joblib.load(os.path.join(cfg["stacking_blueprint_path"], f"{m}.joblib"))
        trained = joblib.load(os.path.join(cfg["stacking_model_path"], f"{m}_trained.joblib"))

        models_dict[m] = {}
        models_dict[m]["blueprint"] = blueprint
        models_dict[m]["trained"] = trained

    logging.info(f"finished loading individual models : {cfg['stacking_models_list']}")
    
    #checking original individual model performances
    logging.info("checking individual model performances...")

    for m in cfg["stacking_models_list"]:
        cv_rmse = cross_val_score(
            models_dict[m]["blueprint"],
            X_train,
            y_train,
            scoring="neg_mean_squared_error",
            cv=5,
        ).mean()
        cv_rmse = (-cv_rmse)**0.5
        logging.info(f"{m} cv rmse : {cv_rmse}")
    
    #evaluating the stacked model
    logging.info("checking cv performance of stacked model...")
    keys = models_dict.keys()
    blueprints = [models_dict[k]['blueprint'] for k in keys]
    estimators = [(k, b) for k, b in zip(keys, blueprints)]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),   # meta-model (can be anything)
        cv=5                       # internal CV for meta-features
    )

    cv_rmse = cross_val_score(
        stacking,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=5,
    ).mean()
    cv_rmse = (-cv_rmse)**0.5

    logging.info(f"stacked model cv rmse : {cv_rmse}")

    #generate test predictions
    logging.info("generate test predictions...")
    stacking.fit(X_train, y_train)
    y_test_preds = stacking.predict(X_test)

    #save predictions
    df_preds = pd.DataFrame({
        "id": df_test["id"],
        target_col: y_test_preds
    })
    preds_save_path = os.path.join(cfg["stacking_output_path"], f"preds/{cfg['stacking_model_name']}_preds.csv")
    df_preds.to_csv(preds_save_path, index=False)
    logging.info(f"stacked model predictions saved to {preds_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)