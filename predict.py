import argparse
import yaml
import joblib
import pandas as pd
import numpy as np

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    #load model
    model = joblib.load(cfg["model_trained_path"])

    #load test data
    test_df = pd.read_csv(cfg["test_data_path"])
    X_test = test_df[cfg["feature_cols"]]

    #predict
    preds = model.predict(X_test)

    #save predictions
    df_preds = pd.DataFrame({
        "id": test_df["id"],
        cfg["target_col"]: preds
    })
    df_preds.to_csv(cfg["preds_path"], index=False)
    print(f"Predictions saved to {cfg['preds_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)