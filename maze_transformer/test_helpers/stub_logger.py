from maze_transformer.training.wandb_logger import WandbLogger


class StubLogger(WandbLogger):
    """Drop-in replacement for the WandbLogger to make it easy to inspect logs during tests (and avoid uploading models and datasets in unit tests)"""

    def __init__(self):
        self.logs = []

    def _log(self, *logs):
        self.logs.append(logs)

    @classmethod
    def create(cls, *args, **kwargs) -> "StubLogger":
        logger = StubLogger()
        logger._log("StubLogger created", args, kwargs)
        return logger

    def upload_model(self, *args, **kwargs) -> None:
        self._log("Model uploaded.", args, kwargs)

    def upload_dataset(self, *args, **kwargs) -> None:
        self._log("Dataset uploaded.", args, kwargs)

    def log_metric(self, *args, **kwargs) -> None:
        self._log("Metric logged.", args, kwargs)

    def log_metric_hist(self, *args, **kwargs) -> None:
        self._log("Metric (Statcounter) logged.", args, kwargs)

    def summary(self, *args, **kwargs) -> None:
        self._log("Summary logged.", args, kwargs)

    def progress(self, message: str) -> None:
        msg: str = f"[INFO] - {message}"
        print(msg)
        self._log(msg)

    @property
    def url(self) -> str:
        return "stub logger, not a url"
