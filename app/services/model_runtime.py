import gc
from contextlib import contextmanager
from threading import RLock
from typing import Generator, Protocol

import torch


class ManagedModel(Protocol):
    def load(self) -> None: ...

    def unload(self) -> None: ...


class ModelRuntime:
    """Keep only the model needed by the current inference request in memory."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._active_model: ManagedModel | None = None

    @contextmanager
    def use(self, model: ManagedModel) -> Generator[None, None, None]:
        with self._lock:
            if self._active_model is not model:
                if self._active_model is not None:
                    self._active_model.unload()
                    self._active_model = None
                    self._release_memory()

                try:
                    model.load()
                except Exception:
                    model.unload()
                    self._release_memory()
                    raise

                self._active_model = model

            yield

    def unload(self) -> None:
        with self._lock:
            if self._active_model is not None:
                self._active_model.unload()
                self._active_model = None
            self._release_memory()

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


model_runtime = ModelRuntime()
