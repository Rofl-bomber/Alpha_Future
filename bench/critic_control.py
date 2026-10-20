"""Контрольный тест Критика: признак из зашумлённого таргета должен быть отклонён.

    [OPENAI_*=...] [AGENT_MOCK_LLM=1] python bench/critic_control.py путь/к/task.md [--noise 0.3]

Строит L1, подкладывает признак «таргет + шум» (в test — случайные значения из того же распределения,
как бывает при реальной утечке), прогоняет эксперимент и отдаёт результат Критику тем же путём,
что и в цикле. Ожидаемый вердикт — reject.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.config import load_config  # noqa: E402
from agent.cv import make_folds  # noqa: E402
from agent.llm.client import LLMClient  # noqa: E402
from agent.llm.mock import MockLLM  # noqa: E402
from agent.llm.roles import Critic, Hypothesis  # noqa: E402
from agent.metrics import improvement, relative_improvement  # noqa: E402
from agent.orchestrator import Orchestrator  # noqa: E402
from agent.pipeline import apply_patch, default_config  # noqa: E402
from agent.profiler import profile  # noqa: E402
from agent.spec import load_data  # noqa: E402

LEAK_CODE = '''import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ref = pd.read_csv("{train}")  # «справочные» значения
    proxy = ref["{target}"].astype(float) + np.random.normal(0, {noise}, len(ref))
    out["usage_index"] = np.concatenate([proxy.values, np.random.choice(proxy.values, len(df) - len(ref))])
    return out
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_md")
    ap.add_argument("--noise", type=float, default=0.3,
                    help="регрессия: std шума в долях RMSE L1; классификация: доля случайно заменённых меток")
    a = ap.parse_args()
    cfg = load_config()
    orch = Orchestrator(a.task_md, cfg)
    spec = orch.spec
    orch.data = load_data(spec, cfg.subsample, cfg.seed)
    orch.prof = profile(spec, orch.data, cfg.profile_max_chars)
    orch.folds = make_folds(spec, orch.data.y, cfg.n_folds, cfg.seed)
    base = orch._run(default_config(spec))
    orch.l1_importances = {k: v for k, v in (base.importances or {}).items() if k in spec.feature_cols}
    base_e = orch.journal.add(None, base, "baseline", iter=0)

    rng = np.random.default_rng(0)
    y = orch.data.y.astype(float)
    if spec.is_clf:
        leak = y.copy()
        flip = rng.random(len(y)) < args_noise_to_flip(a.noise)
        leak[flip] = rng.integers(0, spec.n_classes, flip.sum())
    else:  # шум — доля RMSE L1: признак заметно точнее модели, как и бывает при утечке
        leak = y + rng.normal(0, a.noise * base.score, len(y))
    frame = pd.DataFrame({"usage_index": np.concatenate([leak, rng.choice(leak, orch.data.n_test)])})
    noise_std = 0.0 if spec.is_clf else round(a.noise * base.score, 4)
    code = LEAK_CODE.format(train=spec.train_path.name, target=spec.target, noise=noise_std)
    fe_id = orch.registry.next_id()
    orch.registry.register(fe_id, frame, code, "индекс использования по справочным данным", "fe_usage_index", "control")
    cfg_new = apply_patch(base.config, {"features.custom": [f"+{fe_id}"]})
    res = orch._run(cfg_new)
    delta = improvement(spec.metric, res.score, base.score)
    rel = relative_improvement(spec.metric, res.score, base.score)
    print(f"L1 {base.score:.6g} -> с утечкой {res.score:.6g}; дельта {delta:+.6g} ({rel:+.1%}); "
          f"порог Критика {cfg.jump_threshold:.0%}" + ("" if rel > cfg.jump_threshold else " — ниже порога, Критик вызван принудительно"))

    orch.client = LLMClient(cfg, orch.llm_log, Path(cfg.llm_cache_dir or spec.task_dir / ".agent_cache" / "llm"),
                            mock=MockLLM() if cfg.mock_llm else None)
    orch.critic = Critic(orch.client)
    h = Hypothesis(id="control", kind="feature_code", rationale="индекс использования по справочным данным",
                   patch={"features.custom": ["+fe_usage_index"]}, feature_spec="usage_index из справочных данных",
                   expected="рост метрики")
    hd = h.model_dump()
    verdict = orch._critic_jump(h, hd, res, base_e, base, fe_id, delta, rel)
    ok = verdict["verdict"] == "reject"
    print(f"Связь с таргетом: {verdict['assoc']}")
    print(f"Вердикт Критика: {verdict['verdict']} — {verdict['reason']}")
    print("КОНТРОЛЬ ПРОЙДЕН" if ok else "КОНТРОЛЬ НЕ ПРОЙДЕН: утечка принята")
    print(f"Лог LLM: {orch.llm_log.path}")
    sys.exit(0 if ok else 1)


def args_noise_to_flip(noise: float) -> float:
    """Для классификации шум — доля случайно переставленных меток."""
    return min(0.9, max(0.0, noise))


if __name__ == "__main__":
    main()
