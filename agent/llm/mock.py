"""Подставная LLM для проверки харнесса без эндпоинта и токенов.

Сценарий построен по фазам (C — признаки, D — тюнинг) так, чтобы прогон прошёл все ветки: гипотезы всех типов,
фильтр «не та фаза», плато и смену направления внутри фазы,
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
        if attempt == 0:
            self.n_researcher += 1
        phase = ctx.get("phase", "features")
        k = int(ctx.get("phase_iter", 1))
        if self.n_researcher == 1 and attempt == 0:
            # невалидный JSON: оборванный ответ модели
            return ('Вот гипотезы:\n```json\n{"hypotheses": [{"id": "h1", "kind": "feature_code", '
                    '"rationale": "отношение", "patch": {"features.custom": ["+fe_ratio"]}\n```')
        if phase == "features" and k == 3 and attempt == 0:
            # валидный JSON, но невалидная схема (неизвестный kind)
            return _j({"hypotheses": [{"id": "h9", "kind": "features", "patch": {"features.drop": ["+x"]}}]})
        H = self._features(ctx, k) if phase == "features" else self._tuning(ctx, k)
        banned = set(ctx.get("banned_kinds") or [])
        if banned:
            # одна гипотеза запрещённой категории — проверить фильтр «смены направления»
            H = [h for h in H if h["kind"] in banned][:1] + [h for h in H if h["kind"] not in banned]
        for h in H:
            h.setdefault("feature_spec", None)
            h.setdefault("expected", "")
            h.setdefault("cost", "low")
        return _j({"hypotheses": H[: int(ctx.get("n_hyp", 5)) + 1]})

    @staticmethod
    def _cols(ctx):
        nums = ctx.get("top_num") or ["x0", "x1"]
        a, b = nums[0], nums[1] if len(nums) > 1 else nums[0]
        cats = ctx.get("top_cat") or []
        return nums, a, b, (cats[0] if cats else None)

    def _features(self, ctx: dict, k: int) -> list[dict]:
        nums, a, b, c = self._cols(ctx)
        target = ctx.get("target", "target")

        def fc(hid, alias, spec, why):
            return dict(id=hid, kind="feature_code", rationale=why, patch={"features.custom": [f"+{alias}"]},
                        feature_spec=spec)

        if k == 1:
            return [
                fc("h1", "fe_ratio", f"ratio = {a} / {b}", f"отношение двух самых важных признаков {a}/{b}"),
                fc("h2", "fe_ratio2", f"ratio2 = {a} / ({b} + 1)", f"сглаженное отношение {a}/({b}+1)"),
                dict(id="h3", kind="config", rationale="частоты категорий" if c else "агрегаты по группам",
                     patch={"features.builtin": ["+freq_enc"]} if c else {"features.builtin": ["+group_stats"]}),
                fc("h4", "fe_te", f"mean({target}) по {c or a}", "среднее таргета по категории"),
                dict(id="h5", kind="model", rationale="CatBoost (не та фаза)", patch={"model.name": "catboost"}),
                # невалидная отдельная гипотеза (нет feature_spec) — отфильтруется, остальные пойдут в работу
                dict(id="h5b", kind="feature_code", rationale="забыл спецификацию",
                     patch={"features.custom": ["+fe_x"]}),
            ]
        if k == 2:
            return [
                fc("h6", "fe_broken", f"inter = {a} * частота категории {c or b}", "взаимодействие"),
                fc("h7", "fe_prod", f"prod = {a} * {b}", f"произведение {a} и {b}"),
                dict(id="h8", kind="config", rationale="проверка: без самого важного признака",
                     patch={"features.drop": [f"+{a}"]}),
                dict(id="h9", kind="config", rationale="агрегаты по группам", patch={"features.builtin": ["+group_stats"]}),
            ]
        if k == 3:
            return [
                fc("h10", "fe_sum", f"sum = {a} + {b}", "сумма"),
                fc("h11", "fe_diff", f"diff = {a} - {b}", "разность"),
                fc("h12", "fe_sq", f"sq = {a} * {a}", "квадрат"),
                fc("h13", "fe_half", f"half = {a} / 2", "масштаб"),
                dict(id="h14", kind="config", rationale="скорость обучения (не та фаза)",
                     patch={"model.params.learning_rate": 0.1}),
            ]
        if k == 4:
            return [
                fc("h15", "fe_x", f"x = {a} * 3", "масштаб ещё раз"),
                dict(id="h16", kind="config", rationale="без второго по важности признака",
                     patch={"features.drop": [f"+{b}"]}),
                dict(id="h17", kind="config", rationale="группировка по числовой колонке (ошибка)",
                     patch={"features.builtin": ["+group_stats"], "features.group_by": [a]}),
            ]
        i = min(k - 3, len(nums) - 1)
        return [
            dict(id=f"h{20 + 2 * k}", kind="config", rationale="без двух важных признаков",
                 patch={"features.drop": [f"+{a}", f"+{nums[i]}"]}),
            dict(id=f"h{21 + 2 * k}", kind="config", rationale="без признака", patch={"features.drop": [f"+{nums[i]}"]}),
        ]

    def _tuning(self, ctx: dict, k: int) -> list[dict]:
        m = ctx.get("model_name", "lightgbm")
        clf = ctx.get("task_type") != "regression"
        grid = {
            "lightgbm": [{"learning_rate": 0.03, "num_leaves": 31}, {"num_leaves": 4, "min_child_samples": 300},
                         {"num_leaves": 8, "learning_rate": 0.3}, {"reg_lambda": 50.0}],
            "xgboost": [{"max_depth": 5}, {"max_depth": 2, "min_child_weight": 50},
                        {"max_depth": 3, "learning_rate": 0.3}, {"reg_lambda": 50.0}],
            "catboost": [{"depth": 8}, {"depth": 3, "l2_leaf_reg": 30.0}, {"depth": 4, "learning_rate": 0.3},
                         {"l2_leaf_reg": 50.0}],
        }.get(m, [{}, {}, {}, {}])
        p = grid[min(k - 1, len(grid) - 1)]
        pp = {"postprocess.class_weights_opt": False} if clf else {"postprocess.clip_to_train_range": False}
        other = "xgboost" if m != "xgboost" else "lightgbm"
        H = [dict(id=f"t{k}a", kind="model", rationale=f"параметры {m}", patch={"model.params": p})]
        if k == 1:
            H += [
                dict(id="t1b", kind="postprocess", rationale="проверить вклад постобработки", patch=pp),
                dict(id="t1c", kind="model", rationale="сменить модель (не та фаза)", patch={"model.name": other}),
                dict(id="t1d", kind="model", rationale="то же имя модели — допустимо",
                     patch={"model.name": m, "model.params": {**p, "subsample": 0.7} if m != "catboost" else
                            {**p, "l2_leaf_reg": 5.0}}),
            ]
        elif k == 2:
            H += [dict(id="t2b", kind="postprocess", rationale="повтор постобработки", patch=pp)]
        return H

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
        elif alias == "fe_sum":
            body = f'    out["s"] = df["{a}"] + df["{b}"]\n'
        elif alias == "fe_sq":
            body = f'    out["sq"] = df["{a}"] ** 2\n'
        elif alias == "fe_half":
            body = f'    out["half"] = df["{a}"] / 2.0\n'
        elif alias == "fe_x":
            body = f'    out["x3"] = df["{a}"] * 3.0\n'
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
