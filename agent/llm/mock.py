"""Подставная LLM для проверки харнесса без эндпоинта и токенов.

Сценарий построен так, чтобы прогон прошёл все ветки: гипотезы всех четырёх типов,
невалидный JSON и невалидная схема с повтором, падающий и исправленный код Инженера,
Инженер, не справившийся за 2 исправления, фильтры (таргет, повтор, смена направления),
серия неудач подряд, скачок с вызовом Критика и оба вердикта, финальный выбор.
Ответы строятся из контекста (имена колонок, номер итерации), поэтому mock работает на любой задаче.
"""

from __future__ import annotations

import json


def _usage(messages, text):
    p = sum(len(m.get("content", "")) for m in messages) // 4
    return {"prompt_tokens": p, "completion_tokens": len(text) // 4}


def _j(obj) -> str:
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=1) + "\n```"


class MockLLM:
    def __init__(self):
        self.n_researcher = 0
        self.n_jump = 0

    def complete(self, role: str, messages: list, context: dict, attempt: int):
        if role == "researcher":
            text = self._researcher(context, attempt)
        elif role == "engineer":
            text = self._engineer(context, attempt)
        elif role == "critic":
            text = self._critic(context, attempt)
        else:
            text = "{}"
        return text, _usage(messages, text)

    # ------------------------------------------------------------ Исследователь
    def _researcher(self, ctx: dict, attempt: int) -> str:
        it = int(ctx.get("iteration", 1))
        if attempt == 0:
            self.n_researcher += 1
        nums = ctx.get("top_num") or ["x0", "x1"]
        a, b = nums[0], nums[1] if len(nums) > 1 else nums[0]
        cats = ctx.get("top_cat") or []
        c = cats[0] if cats else None
        clf = ctx.get("task_type") != "regression"
        target = ctx.get("target", "target")
        banned = set(ctx.get("banned_kinds") or [])

        if it == 1 and attempt == 0:
            # невалидный JSON: оборванный ответ модели
            return ('Вот гипотезы:\n```json\n{"hypotheses": [{"id": "h1", "kind": "config", '
                    '"rationale": "флаги пропусков", "patch": {"preprocess.missing_flags": true}\n```')
        if it == 3 and attempt == 0:
            # валидный JSON, но невалидная схема (неизвестный kind)
            return _j({"hypotheses": [{"id": "h9", "kind": "features", "patch": {"features.drop": [f"+{a}"]}}]})

        pp_patch = {"postprocess.class_weights_opt": False} if clf else {"postprocess.clip_to_train_range": False}
        H = []
        if it == 1:
            H = [
                dict(id="h1", kind="config", rationale="флаги пропусков и частоты категорий",
                     patch={"preprocess.missing_flags": True, "features.builtin": ["+freq_enc"]}, expected="небольшой рост"),
                dict(id="h2", kind="feature_code", rationale=f"отношение двух самых важных признаков {a}/{b}",
                     patch={"features.custom": ["+fe_ratio"]}, feature_spec=f"ratio = {a} / {b}",
                     expected="заметный рост"),
                dict(id="h3", kind="model", rationale="CatBoost лучше работает с категориями",
                     patch={"model.name": "catboost"}, expected="рост и разнообразие для бленда"),
                dict(id="h4", kind="postprocess", rationale="проверить вклад постобработки", patch=pp_patch,
                     expected="ухудшение — контроль"),
            ]
        elif it == 2:
            H = [
                dict(id="h5", kind="feature_code", rationale=f"сглаженное отношение {a}/({b}+1) без деления на ноль",
                     patch={"features.custom": ["+fe_ratio2"]}, feature_spec=f"ratio2 = {a} / ({b} + 1)",
                     expected="заметный рост"),
                dict(id="h6", kind="feature_code", rationale="взаимодействие категории и числового признака",
                     patch={"features.custom": ["+fe_broken"]},
                     feature_spec=f"inter = {a} * частота категории {c or b}", expected="рост"),
                dict(id="h7", kind="model", rationale="XGBoost для разнообразия", patch={"model.name": "xgboost"}),
                dict(id="h8", kind="feature_code", rationale="среднее таргета по категории",
                     patch={"features.custom": ["+fe_te"]}, feature_spec=f"mean({target}) по {c or a}"),
            ]
        elif it == 3:
            H = [
                dict(id="h9", kind="config", rationale="проверка: без самого важного признака",
                     patch={"features.drop": [f"+{a}"]}),
                dict(id="h10", kind="config", rationale="все категории в редкие (контроль)",
                     patch={"preprocess.rare_min_count": 10_000_000}),
                dict(id="h11", kind="config", rationale="частотное кодирование вместо категорий",
                     patch={"preprocess.cat_encoding": "freq"}),
                dict(id="h12", kind="config", rationale="без второго по важности признака",
                     patch={"features.drop": [f"+{b}"]}),
                dict(id="h13", kind="config", rationale="агрегаты по группам", patch={"features.builtin": ["+group_stats"]}),
            ]
        elif it == 4:
            H = [
                dict(id="h14", kind="config", rationale="повтор: без самого важного признака",
                     patch={"features.drop": [f"+{a}"]}),
                dict(id="h15", kind="config", rationale="OOF target encoding вместо категорий",
                     patch={"preprocess.cat_encoding": "target"}),
                dict(id="h16", kind="config", rationale="лог-таргет" if not clf else "target encoding в дополнение",
                     patch={"target.transform": "log1p"} if not clf else {"features.builtin": ["+target_enc"]}),
                dict(id="h17", kind="model", rationale="ExtraTrees для бленда", patch={"model.name": "extratrees"}),
                # невалидная отдельная гипотеза (нет feature_spec) — отфильтруется, остальные пойдут в работу
                dict(id="h17b", kind="feature_code", rationale="забыл спецификацию", patch={"features.custom": ["+fe_x"]}),
            ]
        elif it == 5:
            H = [
                dict(id="h18", kind="config", rationale="флаги пропусков ещё раз",
                     patch={"preprocess.missing_flags": False, "preprocess.rare_min_count": 5}),
                dict(id="h19", kind="model", rationale="линейная модель для бленда", patch={"model.name": "linear"}),
                dict(id="h20", kind="model", rationale="MLP для бленда",
                     patch={"model.name": "mlp", "model.params": {"max_epochs": 30}}),
                dict(id="h21", kind="model", rationale="ResNet-MLP для бленда",
                     patch={"model.name": "resnet", "model.params": {"max_epochs": 30}}),
            ]
        elif it == 6:
            H = [
                dict(id="h22", kind="model", rationale="меньше learning rate и листьев",
                     patch={"model.name": "lightgbm", "model.params": {"learning_rate": 0.03, "num_leaves": 31}}),
                dict(id="h23", kind="model", rationale="глубже CatBoost",
                     patch={"model.name": "catboost", "model.params": {"depth": 8}}),
                dict(id="h24", kind="feature_code", rationale=f"произведение {a} и {b}",
                     patch={"features.custom": ["+fe_prod"]}, feature_spec=f"prod = {a} * {b}"),
                dict(id="h25", kind="postprocess", rationale="повтор постобработки", patch=pp_patch),
            ]
        else:
            k = it - 6
            H = [
                dict(id=f"h{30 + 3 * k}", kind="model", rationale="подбор числа листьев",
                     patch={"model.name": "lightgbm", "model.params": {"num_leaves": 15 + 16 * k, "min_child_samples": 10 * k}}),
                dict(id=f"h{31 + 3 * k}", kind="model", rationale="подбор глубины XGBoost",
                     patch={"model.name": "xgboost", "model.params": {"max_depth": 4 + k}}),
                dict(id=f"h{32 + 3 * k}", kind="config", rationale="редкие категории",
                     patch={"preprocess.rare_min_count": 5 * k}),
            ]
        if banned:
            # одна гипотеза запрещённой категории — проверить фильтр «смены направления»
            kept = [h for h in H if h["kind"] not in banned]
            dropped = [h for h in H if h["kind"] in banned][:1]
            H = dropped + kept
        for h in H:
            h.setdefault("feature_spec", None)
            h.setdefault("expected", "")
            h.setdefault("cost", "low")
        return _j({"hypotheses": H[: int(ctx.get("n_hyp", 5))]})

    # ------------------------------------------------------------ Инженер
    def _engineer(self, ctx: dict, attempt: int) -> str:
        alias = ctx.get("alias", "")
        nums = ctx.get("top_num") or ["x0", "x1"]
        a, b = nums[0], nums[1] if len(nums) > 1 else nums[0]
        if alias == "fe_ratio":
            if attempt == 0:  # опечатка в имени колонки -> KeyError (этап exec)
                body = f'    out["ratio"] = df["{a}_total"] / df["{b}"]\n'
            elif attempt == 1:  # деление без защиты от нуля -> inf (этап check)
                body = f'    out["ratio"] = df["{a}"].abs() + 1.0\n    out["ratio"] = out["ratio"] / (df["{b}"] - df["{b}"].min())\n'
            else:
                body = f'    out["ratio"] = df["{a}"] / df["{b}"].replace(0, np.nan)\n'
        elif alias == "fe_ratio2":
            body = f'    out["ratio2"] = df["{a}"] / (df["{b}"].abs() + 1.0)\n'
        elif alias == "fe_broken":
            if attempt < 2:
                body = '    out["inter"] = df["no_such_column"] * 2\n'
            else:
                body = '    out["inter"] = np.nan\n'
        elif alias == "fe_prod":
            body = f'    out["prod"] = df["{a}"] * df["{b}"]\n'
        else:
            body = f'    out["feat"] = df["{a}"] - df["{b}"]\n'
        code = ("import numpy as np\nimport pandas as pd\n\n\ndef make_features(df: pd.DataFrame) -> pd.DataFrame:\n"
                "    out = pd.DataFrame(index=df.index)\n" + body + "    return out\n")
        return "```python\n" + code + "```"

    # ------------------------------------------------------------ Критик
    def _critic(self, ctx: dict, attempt: int) -> str:
        if ctx.get("task") == "jump":
            if attempt == 0:
                self.n_jump += 1
            corr = float(ctx.get("max_abs_corr") or 0.0)
            if self.n_jump == 1:
                return _j({"verdict": "reject", "reason": "Скачок слишком велик для одного отношения, а знаменатель "
                                                         "с нулями даёт пропуски; нужна независимая проверка."})
            if corr > 0.95:
                return _j({"verdict": "reject", "reason": f"Признак почти повторяет таргет (|corr|={corr:.3f}) — утечка."})
            return _j({"verdict": "accept", "reason": f"Прирост объясним смыслом признака, |corr|={corr:.3f}, "
                                                     "улучшение на всех фолдах."})
        # финальный выбор
        gib = bool(ctx.get("greater_is_better"))
        bs, ss = ctx.get("blend_score"), ctx.get("best_single_score")
        if bs is not None and ss is not None and ((bs > ss) if gib else (bs < ss)):
            return _j({"choice": "blend", "reason": "Бленд лучше лучшей одиночной модели по OOF."})
        return _j({"choice": ctx.get("best_single_id", "blend"), "reason": "Бленд не лучше одиночной модели."})
