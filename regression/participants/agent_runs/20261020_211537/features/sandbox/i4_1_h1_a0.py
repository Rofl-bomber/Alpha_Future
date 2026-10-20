import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    star = df["energy_star_rating"].astype(float)
    area = df["floor_area"].astype(float)

    star = star.mask(~np.isfinite(star), np.nan)
    area = area.mask(~np.isfinite(area), np.nan)

    area_safe = area.mask(area <= -1, np.nan)
    log_area = np.log1p(area_safe)

    with np.errstate(over="ignore", invalid="ignore"):
        vals = star * log_area

    vals = vals.mask(~np.isfinite(vals), np.nan)

    return pd.DataFrame({"fe_star_area_log": vals}, index=df.index)