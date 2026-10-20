import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    year = df["year_built"].astype(float)
    hdd = df["heating_degree_days_c"].astype(float)
    cdd = df["cooling_degree_days_c"].astype(float)

    valid = (
        year.notna()
        & (year > 0)
        & np.isfinite(year)
        & np.isfinite(hdd)
        & np.isfinite(cdd)
    )

    safe_year = year.where(valid, 0.0)
    safe_hdd = hdd.where(valid, 0.0)
    safe_cdd = cdd.where(valid, 0.0)

    out = (2015.0 - safe_year) * (safe_hdd + safe_cdd)
    out = out.where(valid, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan)

    if out.isna().all():
        out = out.fillna(0.0)

    return pd.DataFrame({"fe_building_age_total_dd": out}, index=df.index)