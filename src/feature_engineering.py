import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class OutlierRemoval(BaseEstimator, TransformerMixin):
    """
    Outlier removal for sklearn pipelines
    """
    def __init__(self, k=1.5):
        self.k=k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        self.q1 = X.quantile(0.25)
        self.q3 = X.quantile(0.75)
        self.iqr = self.q3 - self.q1
        
        self.median = X.quantile(0.50)

        return self
        
    def transform(self, X):

        lower_bound = self.q1 - self.k * self.iqr
        upper_bound = self.q3 + self.k * self.iqr
        mask = (X < lower_bound) | (X > upper_bound)
        X[mask] = np.nan

        return X.fillna(self.median)  # replace removed values with median
    
def get_custom_pipeline(
    numeric_features,
    regressor_model,
    outlier=True,
    scaler=True,
    poly=True,
    binning=True,
    pca=True,
):
    steps = []
    if outlier:
        steps.append(("outlier", OutlierRemoval()))
    if scaler:
        steps.append(("scaler", StandardScaler()))
    if poly:
        steps.append(("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)))
    if binning:
        steps.append(("binning", KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile", quantile_method="linear")))
    if pca:
        #apply PCA for dimension reduction
        #retain 95% of variance
        steps.append(("pca", PCA(n_components=0.95)))

    numeric_transformer = Pipeline(steps=steps)

    # Apply preprocessing to all numeric features
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )

    # Final pipeline with model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor_model)
    ])

    return model