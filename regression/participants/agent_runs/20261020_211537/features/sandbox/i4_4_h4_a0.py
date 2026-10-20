import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    diff = (df["heating_degree_days_c"] - df["cooling_degree_days_c"]).astype(float)
    diff = diff.mask(diff.isinf(), np.nan)

    if diff.isna().all():
        diff = diff.fillna(0.0)

    return pd.DataFrame({"fe_hdd_cdd_diff": diff}, index=df.index)