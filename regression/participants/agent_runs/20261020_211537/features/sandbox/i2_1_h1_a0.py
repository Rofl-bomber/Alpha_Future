import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    hdd = df["heating_degree_days_c"].astype(float)
    cdd = df["cooling_degree_days_c"].astype(float)

    denom = cdd + 1.0
    denom = denom.mask(denom == 0, np.nan)

    ratio = hdd / denom
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    if ratio.isna().all():
        ratio = ratio.fillna(0.0)

    return pd.DataFrame({"fe_hdd_cdd_ratio": ratio}, index=df.index)