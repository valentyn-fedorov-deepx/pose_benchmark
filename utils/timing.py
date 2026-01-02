import time


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._t0 = None
        self.elapsed = 0.0

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        if self._t0 is None:
            return
        self.elapsed += time.perf_counter() - self._t0
        self._t0 = None

    def get(self):
        return self.elapsed
