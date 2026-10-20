import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df["july_max_temp_c"] - df["january_min_temp_c"]
    feat = feat.replace([np.inf, -np.inf], np.nan)

    if feat.isna().all():
        feat = feat.fillna(0.0)

    return pd.DataFrame({"fe_temp_range": feat}, index=df.index)