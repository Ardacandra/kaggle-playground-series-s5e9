import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, KBinsDiscretizer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge, ARDRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
import os
import joblib
import argparse
import yaml

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.feature_engineering import *
from src.modeling import *

import warnings
warnings.filterwarnings("ignore")

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    #for easier iterations, just hardcode the parameters and model definition in this script
    OUTPUT_PATH = "output/blueprint/"
    BLUEPRINT_NAME = "robust_bayesian_alpha_1_1e_-06_alpha_2_0_01_lambda_1_0_01_lambda_2_0_01.joblib"

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    steps = []

    # steps.append(("outlier_removal", OutlierRemoval()))
    steps.append((("robust_scaler"), RobustScaler()))
    # steps.append(("polynomial", PolynomialFeatures(interaction_only=False, include_bias=False, degree=3)))
    # steps.append(("binning", KBinsDiscretizer(encode="ordinal", n_bins=5)))
    # steps.append(("standardization", StandardScaler()))
    # steps.append(("pca", PCA(n_components=0.85)))

    # steps.append(("model", Lasso(alpha=0.001, max_iter=1000)))
    # steps.append(("model", Ridge(alpha=0.001, solver="auto")))
    steps.append(("model", BayesianRidge(alpha_1=1e-06, alpha_2=0.01, lambda_1=0.01, lambda_2=0.01)))

    pipeline = Pipeline(steps=steps)

    #save best estimator to output folder
    joblib.dump(pipeline, os.path.join(OUTPUT_PATH, BLUEPRINT_NAME))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)