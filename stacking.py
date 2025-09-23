import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
import itertools

import os
import joblib
import argparse
import yaml
import logging

import warnings
warnings.filterwarnings("ignore")

class CustomStackingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimators, final_estimator, n_folds=5, shuffle=True, random_state=None, search_combinations=True):
        self.base_estimators = base_estimators  # list of (name, estimator)
        self.final_estimator = final_estimator
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.search_combinations = True

    def fit(self, X, y):
        if self.search_combinations:
            best_score = np.inf
            best_combo = None

            # generate all non-empty subsets of base estimators
            for r in range(1, len(self.base_estimators) + 1):
                for combo in itertools.combinations(self.base_estimators, r):
                    score = self._evaluate_combo(combo, X, y)
                    if score < best_score:
                        best_score = score
                        best_combo = combo

            self.selected_estimators_ = [clone(est) for _, est in best_combo]
        else:
            # store fitted base estimators
            self.selected_estimators_ = [clone(est) for _, est in self.base_estimators]

        # fit meta-features with the selected combo
        self.meta_features_ = self._fit_estimators(self.selected_estimators_, X, y)

        # fit meta-learner
        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(self.meta_features_, y)

        return self


    def _evaluate_combo(self, combo, X, y):
        """Evaluate a subset of estimators with CV RMSE on meta-features."""
        meta_features = self._fit_estimators([clone(est) for _, est in combo], X, y, refit=False)

        # cross-validate final estimator on the meta features
        scores = cross_val_score(
            clone(self.final_estimator),
            meta_features, y,
            cv=self.n_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        return -scores.mean()

    def _fit_estimators(self, estimators, X, y, refit=True):
        """Fit base estimators with out-of-fold predictions to create meta-features."""
        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(estimators)))

        for i, est in enumerate(estimators):
            oof_preds = np.zeros(X.shape[0])
            for train_idx, val_idx in kf.split(X, y):
                est_clone = clone(est)
                est_clone.fit(X.iloc[train_idx].copy(), y.iloc[train_idx].copy())
                preds = est_clone.predict(X.iloc[val_idx].copy())
                oof_preds[val_idx] = preds.ravel()
            meta_features[:, i] = oof_preds
            if refit:
                est.fit(X.copy(), y.copy())

        return meta_features

    def predict(self, X):

        # get predictions from base learners
        base_preds = np.column_stack([est.predict(X.copy()) for est in self.selected_estimators_])

        # meta-learner makes final prediction
        return self.final_estimator_.predict(base_preds)

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
        # trained = joblib.load(os.path.join(cfg["stacking_model_path"], f"{m}_trained.joblib"))

        models_dict[m] = {}
        models_dict[m]["blueprint"] = blueprint
        # models_dict[m]["trained"] = trained

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
            n_jobs=-1,
        ).mean()
        cv_rmse = (-cv_rmse)**0.5
        logging.info(f"{m} cv rmse : {cv_rmse}")
    
    #evaluating the stacked model
    logging.info("checking cv performance of stacked model...")
    keys = models_dict.keys()
    blueprints = [models_dict[k]['blueprint'] for k in keys]
    estimators = [(k, b) for k, b in zip(keys, blueprints)]

    # stacking = StackingRegressor(
    #     estimators=estimators,
    #     final_estimator=Ridge(alpha=1.0),   # meta-model (can be anything)
    #     cv=5                       # internal CV for meta-features
    # )
    stacking = CustomStackingRegressor(
        base_estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        n_folds=3,
        random_state=42,
        search_combinations=True,
    )

    cv_rmse = cross_val_score(
        stacking,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
    ).mean()
    cv_rmse = (-cv_rmse)**0.5

    logging.info(f"stacked model cv rmse : {cv_rmse}")

    #generate test predictions
    logging.info("train final stacking model...")
    stacking.fit(X_train, y_train)
    model_save_path = os.path.join(cfg["stacking_model_path"], f"{cfg['stacking_model_name']}.joblib")
    joblib.dump(stacking, model_save_path)
    logging.info(f"trained final stacking model saved to {model_save_path}")

    logging.info("generate test predictions...")
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