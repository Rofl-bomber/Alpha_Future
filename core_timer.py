"""Часы агента, реестр дочерних процессов и watchdog.

Отсчёт идёт от старта solution.py (t0 передаётся из него). Фазы задаются долями
общего бюджета (core_config.PHASES). Watchdog в момент жёсткого дедлайна убивает
дочерние процессы и завершает агента: лучший сабмит к этому моменту уже на диске,
потому что каждая запись атомарная.
"""

import logging
import os
import signal
import threading
import time

log = logging.getLogger("agent")


class Timer:
    def __init__(self, t0: float, budget: float, phases: dict):
        self.t0 = t0
        self.budget = float(budget)
        self.phases = dict(phases)

    def now(self) -> float:
        return time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float:
        return self.budget - self.elapsed()

    def deadline(self, phase: str) -> float:
        """Абсолютное время (epoch) конца фазы."""
        return self.t0 + self.budget * self.phases[phase]

    def left(self, phase: str) -> float:
        """Секунд до конца фазы (может быть отрицательным)."""
        return self.deadline(phase) - time.time()

    def hard_deadline(self) -> float:
        return self.t0 + self.budget

    def stamp(self) -> str:
        e = self.elapsed()
        return f"{int(e // 60):3d}:{int(e % 60):02d}"


class ProcessRegistry:
    """Живые дочерние процессы: watchdog и таймауты убивают их вместе с потомками."""

    def __init__(self):
        self._procs = {}
        self._lock = threading.Lock()

    def add(self, proc):
        with self._lock:
            self._procs[proc.pid] = proc

    def remove(self, proc):
        with self._lock:
            self._procs.pop(proc.pid, None)

    def kill_all(self):
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            kill_tree(proc.pid)


def kill_tree(pid: int):
    """Убить процесс и всех его потомков; группа процессов создаётся при запуске (start_new_session)."""
    try:
        import psutil

        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            parent, children = None, []
        for child in children + ([parent] if parent is not None else []):
            try:
                child.kill()
            except Exception:
                pass
    except Exception:
        pass
    sig = getattr(signal, "SIGKILL", signal.SIGTERM)  # на Windows SIGKILL нет; SIGTERM = TerminateProcess
    if hasattr(os, "killpg"):
        try:
            os.killpg(pid, sig)
        except Exception:
            pass
    try:
        os.kill(pid, sig)
    except Exception:
        pass


class Watchdog(threading.Thread):
    """Следит за жёстким дедлайном. По его наступлении убивает дочерние процессы и выходит."""

    def __init__(self, timer: Timer, registry: ProcessRegistry, on_exit=None):
        super().__init__(daemon=True, name="watchdog")
        self.timer = timer
        self.registry = registry
        self.on_exit = on_exit
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            left = self.timer.hard_deadline() - time.time()
            if left <= 0:
                break
            self._stop.wait(min(1.0, max(0.05, left)))
        if self._stop.is_set():
            return
        try:
            log.warning("Watchdog: жёсткий дедлайн, останавливаю дочерние процессы")
            self.registry.kill_all()
            if self.on_exit:
                self.on_exit()
        finally:
            try:
                logging.shutdown()
            finally:
                os._exit(0)
