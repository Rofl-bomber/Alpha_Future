import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    total = df["heating_degree_days_c"].add(df["cooling_degree_days_c"])
    total = total.where(np.isfinite(total), np.nan)

    if total.isna().all():
        total = total.fillna(0.0)

    return pd.DataFrame({"fe_total_dd": total}, index=df.index)