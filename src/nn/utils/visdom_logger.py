from __future__ import annotations

from typing import Any, Dict, Optional

from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning_utilities.core.rank_zero import rank_zero_only


class VisdomLogger(Logger):
    """Minimal Visdom logger compatible with Lightning."""

    def __init__(
        self,
        save_dir: Optional[str] = None,
        name: Optional[str] = None,
        server: str = "http://localhost",
        port: int = 8097,
        env: str = "main",
        raise_exceptions: bool = True,
    ) -> None:
        super().__init__()
        self._save_dir = save_dir
        self._name = name or "visdom"
        self._server = server
        self._port = port
        self._env = env
        self._raise_exceptions = raise_exceptions
        self._viz = None
        self._metric_windows: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._env

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._viz is None:
            try:
                import visdom
            except Exception as exc:
                raise ImportError(
                    "Visdom is not installed. Install it with `pip install visdom`."
                ) from exc
            self._viz = visdom.Visdom(
                server=self._server,
                port=self._port,
                env=self._env,
                raise_exceptions=self._raise_exceptions,
            )
        return self._viz

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        # Visdom doesn't have a native hparams panel; no-op for now.
        _ = params

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if step is None:
            step = 0
        for name, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue

            if name in self._metric_windows:
                self.experiment.line(
                    X=[step],
                    Y=[value],
                    win=self._metric_windows[name],
                    update="append",
                )
            else:
                win = self.experiment.line(
                    X=[step],
                    Y=[value],
                    opts={"title": name, "xlabel": "step", "ylabel": name},
                )
                self._metric_windows[name] = win

    @rank_zero_only
    def finalize(self, status: str) -> None:
        _ = status
