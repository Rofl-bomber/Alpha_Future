
import sys, traceback, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
code_path, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    df = pd.read_pickle(in_path)
    ns = {"np": np, "pd": pd, "__name__": "fe_module"}
    src = open(code_path, encoding="utf-8").read()
    exec(compile(src, "make_features.py", "exec"), ns)
    if "make_features" not in ns:
        raise NameError("в коде нет функции make_features(df)")
    res = ns["make_features"](df.copy())
    if isinstance(res, pd.Series):
        res = res.to_frame()
    if not isinstance(res, pd.DataFrame):
        raise TypeError(f"make_features должна вернуть pandas.DataFrame, получено {type(res).__name__}")
    res.to_pickle(out_path)
except BaseException:
    sys.stderr.write(traceback.format_exc(limit=8))
    sys.exit(3)
