import argparse
import yaml
import joblib
import pandas as pd
import numpy as np

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    #load data
    df = pd.read_csv(cfg["train_data_path"])
    X = df[cfg["feature_cols"]]
    y = df[cfg["target_col"]]

    #train model
    model = joblib.load(cfg["model_blueprint_path"])
    model.fit(X, y)

    #save model
    joblib.dump(model, cfg["model_trained_path"])
    print(f"Model training finished. Model saved to {cfg['model_trained_path']}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)