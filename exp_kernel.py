"""Постоянная Python-сессия (как ноутбук) для Аналитика и Исследователя.

Родитель (Kernel) запускает `python exp_kernel.py conf.json` в отдельной группе процессов
(виден watchdog через ProcessRegistry), без ключей OPENAI_* в окружении и с ограничением потоков.
Запросы идут JSON-строкой в stdin, ответы — JSON-строкой в stdout. Ребёнок сразу уводит свой
дескриптор 1 в лог (stderr), а для протокола оставляет его копию, поэтому вывод кода пользователя
и библиотек не смешивается с протоколом. Работает и на Linux, и на Windows. Сессия живёт весь прогон.

В пространстве имён: train, test, data_dictionary, sample_submission (сырые таблицы), pd, np и
функция харнесса submit_split. Харнесс подгружает туда же текущее решение (load).

Механика шага (исправления по прогонам eda_probe):
- выполняется код одного шага; если последняя строка — выражение, печатается его значение;
- stdout и stderr перехватываются, warnings выключены, длинный вывод сокращается до начала и конца;
- train / test и другие таблицы харнесса защищены: если шаг их подменил или изменил колонки
  и форму, они восстанавливаются из эталона, а в вывод добавляется предупреждение;
- таймаут шага прерывает код (SIGALRM, на Windows — прерывание из потока-таймера), сессия и переменные сохраняются; если код не реагирует,
  родитель убивает процесс и запускает сессию заново (переменные теряются, об этом сказано в выводе).
"""

import ast
import contextlib
import io
import _thread
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTECTED = ("train", "test", "data_dictionary", "sample_submission")


# =====================================================================================
# Родитель
# =====================================================================================


class Kernel:
    def __init__(self, ctx, conf: dict, log=None):
        self.ctx = ctx
        self.conf = dict(conf)
        self.log = log
        self.proc = None
        self.q = None  # строки протокола из stdout ребёнка (читает поток)
        self.lock = threading.Lock()
        self.loaded = {}  # что подгружено харнессом: после перезапуска подгружается снова
        self.restarts = 0
        self.code_path = None  # код текущего шага пишется сюда (submit_split сохраняет его)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout: float = 600.0) -> dict:
        import exp_runner

        kdir = self.ctx.p("kernel", "conf.json").parent
        conf_path = kdir / "conf.json"
        conf_path.write_text(json.dumps(self.conf, ensure_ascii=False, default=str), encoding="utf-8")
        log_path = self.ctx.p("logs", "kernel.log")
        lf = open(log_path, "a", encoding="utf-8")
        env = exp_runner.child_env(int(self.conf.get("threads", 2)))
        self.proc = subprocess.Popen([sys.executable, "-u", str(HERE / "exp_kernel.py"), str(conf_path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=lf,
                                     env=env, cwd=str(self.ctx.workdir), start_new_session=True)
        lf.close()
        self.q = queue.Queue()
        threading.Thread(target=_pump, args=(self.proc.stdout, self.q), daemon=True).start()
        self.ctx.registry.add(self.proc)
        res = self._read(timeout)
        if not res.get("ok"):
            self.close()
            raise RuntimeError(f"Python-сессия не запустилась: {res.get('error')}")
        if self.loaded:
            self._send({"op": "load", "items": self.loaded})
            self._read(300.0)
        return res

    def _send(self, obj):
        self.proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _read(self, timeout: float) -> dict:
        t_end = time.time() + timeout
        while True:
            left = t_end - time.time()
            if left <= 0:
                return {"ok": False, "timeout": True, "error": "нет ответа сессии"}
            try:
                line = self.q.get(timeout=min(left, 1.0))
            except queue.Empty:
                if not self.alive and self.q.empty():
                    return {"ok": False, "dead": True, "error": "процесс сессии завершился"}
                continue
            if line is None:
                return {"ok": False, "dead": True, "error": "процесс сессии завершился"}
            try:
                return json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    def request(self, obj: dict, timeout: float) -> dict:
        with self.lock:
            if not self.alive:
                self._restart_locked()
            try:
                self._send(obj)
            except (BrokenPipeError, OSError):
                self._restart_locked()
                self._send(obj)
            res = self._read(timeout)
            if res.get("timeout") or res.get("dead"):
                why = "не ответила вовремя" if res.get("timeout") else "упала"
                self._restart_locked()
                res = {"ok": False, "restarted": True,
                       "output": f"TIMEOUT/CRASH: сессия {why} и была перезапущена. Все переменные потеряны; "
                                 f"train, test и данные харнесса загружены заново."}
            return res

    def _restart_locked(self):
        from core_timer import kill_tree

        if self.proc is not None:
            kill_tree(self.proc.pid)
            self.ctx.registry.remove(self.proc)
            _close_pipes(self.proc)
        self.restarts += 1
        if self.log:
            self.log.warning("Python-сессия перезапущена (%d-й раз)", self.restarts)
        self.start()

    def exec(self, code: str, timeout: float | None = None, label: str = "") -> str:
        """Выполнить код шага; вернуть вывод (уже сокращённый)."""
        t = float(timeout or self.conf.get("step_timeout", 120.0))
        res = self.request({"op": "exec", "code": code, "timeout": t, "label": label}, timeout=t + 30.0)
        return res.get("output", "") if res.get("output") is not None else str(res.get("error", ""))

    def load(self, items: dict, timeout: float = 300.0) -> dict:
        """Подгрузить в сессию таблицы и массивы харнесса: {имя: путь к .pkl или .npy}."""
        self.loaded.update(items)
        return self.request({"op": "load", "items": items}, timeout=timeout)

    def close(self):
        from core_timer import kill_tree

        if self.proc is not None:
            try:
                self._send({"op": "exit"})
            except Exception:
                pass
            kill_tree(self.proc.pid)
            self.ctx.registry.remove(self.proc)
            _close_pipes(self.proc)
            self.proc = None


def _pump(stream, q):
    """Поток родителя: строки протокола из stdout ребёнка в очередь; None — конец потока."""
    try:
        for line in iter(stream.readline, b""):
            q.put(line)
    except Exception:
        pass
    q.put(None)


def _close_pipes(proc):
    for f in (proc.stdin, proc.stdout):
        try:
            f.close()
        except Exception:
            pass


# =====================================================================================
# Ребёнок
# =====================================================================================


class StepTimeout(BaseException):
    pass


TIMEOUT_MSG = ("\nTIMEOUT: код выполнялся дольше {timeout:.0f} с и был прерван. "
               "Переменные сессии сохранены; ускорьте код (подвыборка, векторизация).")


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return (text[:head] + f"\n... [вывод сокращён: вырезано {len(text) - limit} символов] ...\n" + text[-tail:])


def _snapshot(obj):
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        return ("df", obj.shape, tuple(map(str, obj.columns)), tuple(map(str, obj.dtypes)))
    return ("obj", type(obj).__name__, getattr(obj, "shape", None))


class Session:
    def __init__(self, conf: dict):
        import numpy as np
        import pandas as pd

        import task_spec

        self.conf = conf
        self.np, self.pd = np, pd
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 60)
        pd.set_option("display.max_colwidth", 40)
        paths = conf["paths"]
        train = task_spec.read_table(paths["train"])
        test = task_spec.read_table(paths["test"])
        dd = task_spec.read_table(paths["dict"]) if paths.get("dict") else None
        sample = task_spec.read_table(paths["sample"]) if paths.get("sample") else None
        self.ref = {"train": train, "test": test, "data_dictionary": dd, "sample_submission": sample}
        self.ns = {"pd": pd, "np": np, "__name__": "__session__"}
        for k, v in self.ref.items():
            self.ns[k] = v.copy() if v is not None else None
        self.guard = {}
        for k in PROTECTED:
            self._protect(k, self.ns[k])
        self.ns["submit_split"] = self.submit_split
        self.code = ""

    # ---------------------------------------------------------------- защита данных
    def _protect(self, name, obj):
        self.guard[name] = (obj, _snapshot(obj) if obj is not None else None)

    def _check_guard(self) -> list:
        notes = []
        for name, (obj, snap) in list(self.guard.items()):
            cur = self.ns.get(name)
            if obj is None:
                continue
            changed = cur is not obj or _snapshot(cur) != snap
            if not changed:
                continue
            ref = self.ref.get(name)
            restored = ref.copy() if hasattr(ref, "copy") else ref
            what = ""
            if hasattr(cur, "columns") and hasattr(obj, "columns"):
                added = [c for c in cur.columns if c not in set(map(str, snap[2]))] if snap else []
                what = f" (добавлены колонки {added[:8]})" if added else ""
            self.ns[name] = restored
            self._protect(name, restored)
            notes.append(f"WARNING: `{name}` был изменён шагом{what} — восстановлен исходный. "
                         f"Работайте с копией: df = {name}.copy().")
        return notes

    # ---------------------------------------------------------------- шаг
    def run(self, code: str, timeout: float) -> str:
        buf = io.StringIO()
        self.code = code

        fired = []
        use_alarm = hasattr(signal, "setitimer")
        timer = None

        def on_alarm(signum, frame):
            raise StepTimeout()

        def on_timer():  # Windows: SIGALRM нет — прерываем главный поток
            fired.append(1)
            _thread.interrupt_main()

        if use_alarm:
            signal.signal(signal.SIGALRM, on_alarm)
            signal.setitimer(signal.ITIMER_REAL, max(1.0, float(timeout)))
        else:
            timer = threading.Timer(max(1.0, float(timeout)), on_timer)
            timer.daemon = True
            timer.start()
        try:
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    tree = ast.parse(code, "<step>", "exec")
                    last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                    body = tree.body[:-1] if last is not None else tree.body
                    exec(compile(ast.Module(body=body, type_ignores=[]), "<step>", "exec"), self.ns)
                    if last is not None:
                        v = eval(compile(ast.Expression(last.value), "<step>", "eval"), self.ns)
                        if v is not None:
                            print(repr(v))
            finally:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                else:
                    timer.cancel()
        except StepTimeout:
            buf.write(TIMEOUT_MSG.format(timeout=timeout))
        except KeyboardInterrupt:
            buf.write(TIMEOUT_MSG.format(timeout=timeout) if fired else "\nKeyboardInterrupt")
        except SyntaxError as e:
            buf.write(f"\nSyntaxError: {e.msg} (строка {e.lineno}): {(e.text or '').strip()}")
        except BaseException as e:  # noqa: BLE001
            buf.write("\n" + _format_error(e))
        out = buf.getvalue()
        notes = self._check_guard()
        if notes:
            out = out.rstrip() + "\n" + "\n".join(notes)
        if not out.strip():
            out = "(нет вывода: используйте print())"
        return _cut(out, int(self.conf.get("max_out", 4000)))

    def load(self, items: dict) -> str:
        import numpy as np
        import pandas as pd

        done = []
        for name, path in items.items():
            p = str(path)
            if p.endswith(".npy"):
                obj = np.load(p, allow_pickle=False)
            else:
                obj = pd.read_pickle(p)
            self.ns[name] = obj
            self.ref[name] = obj.copy() if hasattr(obj, "copy") else obj
            self._protect(name, obj)
            done.append(name)
        return "загружено: " + ", ".join(done)

    # ---------------------------------------------------------------- функция харнесса
    def submit_split(self, holdout, folds=None, type="stratified", columns=None, n_folds=5):
        """Зафиксировать разбиение: holdout (маска строк train или их id), фолды по типу или свои."""
        import numpy as np
        import pandas as pd

        import eval_protocol as ep

        conf = self.conf
        T = self.ref["train"]
        if isinstance(columns, str):
            columns = [columns]
        columns = list(columns or [])
        kind = str(type or "stratified").lower()
        try:
            if kind not in ep.SPLIT_TYPES:
                raise ValueError(f"type должен быть одним из {ep.SPLIT_TYPES}")
            bad = [c for c in columns if c not in T.columns]
            if bad:
                raise ValueError(f"колонок {bad} нет в train")
            hm = ep.resolve_rows(holdout, T, conf.get("id_col"))
            targets = conf["targets"]
            valid = T[targets].notna().all(axis=1).to_numpy() if conf["task_type"] in (
                "regression", "multi_regression", "multilabel") else T[targets[0]].notna().to_numpy()
            y_valid = T.loc[valid, targets[0]]
            search = valid & ~hm
            is_cls = conf["task_type"] in ("binary", "multiclass")
            y_s = T.loc[search, targets[0]]
            y_enc = pd.factorize(y_s.astype(str))[0] if is_cls else pd.to_numeric(y_s, errors="coerce").to_numpy()
            alt_raw = None
            if folds is None:
                if kind == "custom":
                    raise ValueError("для type='custom' передайте folds — номер фолда для каждой строки train")
                frame = T.loc[search].reset_index(drop=True)
                k = int(n_folds)
                fs = ep.build_folds(kind, frame, y_enc, conf["task_type"], columns, k, int(conf.get("seed", 42)))
                folds_raw = np.full(len(T), -1, int)
                folds_raw[search] = fs
                if kind in ("random", "stratified", "group"):
                    fa = ep.build_folds(kind, frame, y_enc, conf["task_type"], columns, k,
                                        int(conf.get("seed", 42)) + 7)
                    alt_raw = np.full(len(T), -1, int)
                    alt_raw[search] = fa
            else:
                fr = folds.to_numpy() if isinstance(folds, (pd.Series, pd.Index)) else np.asarray(folds)
                if fr.shape[0] != len(T):
                    raise ValueError(f"folds длины {fr.shape[0]}, а строк в train {len(T)}")
                fr = pd.to_numeric(pd.Series(fr), errors="coerce").fillna(-1).to_numpy().astype(int)
                folds_raw = np.where(search, fr, -1)
            errors, summary = ep.check_split(hm, folds_raw, valid, y_valid, conf["task_type"],
                                             int(conf.get("n_classes", 0)), float(conf["holdout_min"]),
                                             float(conf["holdout_max"]))
        except Exception as e:  # noqa: BLE001
            print(f"submit_split ОТКЛОНЁН: {e.__class__.__name__}: {e}")
            return None
        if errors:
            print("submit_split ОТКЛОНЁН:\n- " + "\n- ".join(errors) + "\n" + summary)
            return None
        wd = Path(conf["workdir"])
        ep.save_analyst_split(wd, hm, folds_raw, alt_raw, {"type": kind, "columns": columns, "n_folds": int(n_folds)})
        (wd / "analyst_split_code.py").write_text(self.code, encoding="utf-8")
        print("submit_split ПРИНЯТ (используется харнессом; повторный вызов заменит его):\n" + summary)
        return None


def _format_error(e: BaseException) -> str:
    tb = traceback.extract_tb(e.__traceback__)
    user = [fr for fr in tb if fr.filename == "<step>"]
    where = f" (строка {user[-1].lineno} вашего кода)" if user else ""
    tail = [fr for fr in tb if fr.filename != "<step>" and not fr.filename.endswith("exp_kernel.py")][-2:]
    lines = [f"{type(e).__name__}{where}: {e}"]
    for fr in tail:
        lines.append(f"  в {Path(fr.filename).name}:{fr.lineno} {fr.name}")
    return "\n".join(lines)


def main():
    import warnings

    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    conf = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # Канал протокола — копия исходного stdout; сам дескриптор 1 уводим в лог (stderr), чтобы
    # печать из C-кода и библиотек не попадала в протокол.
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    try:
        sys.stdout.flush()
        os.dup2(2, 1)
    except OSError:
        pass
    if not hasattr(signal, "setitimer"):  # Windows: таймаут шага идёт через interrupt_main
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except Exception:
            pass

    def send(obj):
        out.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        out.flush()

    try:
        sess = Session(conf)
    except Exception as e:  # noqa: BLE001
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return
    send({"ok": True, "ready": True})
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                send({"ok": False, "error": "неразборчивый запрос"})
                continue
            op = req.get("op")
            if op == "exit":
                break
            try:
                if op == "exec":
                    send({"ok": True, "output": sess.run(req.get("code", ""), float(req.get("timeout", 120)))})
                elif op == "load":
                    send({"ok": True, "output": sess.load(req.get("items", {}))})
                elif op == "ping":
                    send({"ok": True, "output": "pong"})
                else:
                    send({"ok": False, "error": f"неизвестная операция {op}"})
            except KeyboardInterrupt:
                raise
            except BaseException as e:  # noqa: BLE001
                send({"ok": False, "error": f"{type(e).__name__}: {e}"})
        except KeyboardInterrupt:  # запоздалое прерывание таймера между шагами — не выходим
            continue


if __name__ == "__main__":
    main()
