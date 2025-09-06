import numpy as np
import pandas as pd

def calculate_psi(ref, new, bins=10):
    """
    Calculate PSI (Population Stability Index) for numeric features.

    Parameters
    ----------
    ref : list
        Baseline values.
    new : list
        New values to compare against.
    bins : int
        Number of bins to use for quantile-based binning.

    Returns
    -------
    float
        PSI value
    """
    ref = pd.Series(ref)
    new = pd.Series(new)

    # Bin edges based on reference quantiles
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = ref.quantile(quantiles).values

    # Handle duplicate breakpoints (constant features)
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 1:
        return 0 

    # Bin train and test
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    new_counts, _ = np.histogram(new, bins=breakpoints)

    # Convert counts to proportions
    ref_percents = ref_counts / len(ref)
    new_percents = new_counts / len(new)

    # Replace 0s with small number to avoid div/0
    ref_percents = np.where(ref_percents == 0, 1e-6, ref_percents)
    new_percents = np.where(new_percents == 0, 1e-6, new_percents)

    # PSI calculation
    psi = np.sum((ref_percents - new_percents) * np.log(ref_percents / new_percents))
    
    return psi