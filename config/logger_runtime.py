"""标准库 logging 封装：控制台 INFO + 持续写入 rag.log（每 5 小时轮转）。"""
import logging
from datetime import datetime
from pathlib import Path

MAX_LOG_FILES = 50
RUNTIME_LOG_NAME = "rag.log"
SEGMENT_META_NAME = "rag.log.segment"
ROTATE_HOURS = 5


def _is_archive_log(path: Path) -> bool:
    """是否为历史归档日志（时间戳文件），排除 rag.log。"""
    return path.is_file() and path.suffix == ".log" and path.name != RUNTIME_LOG_NAME


def _cleanup_old_logs(log_dir: Path, keep: int = MAX_LOG_FILES) -> None:
    """仅保留最新 keep 个归档日志，rag.log 不参与容量统计。"""
    try:
        files = sorted((p for p in log_dir.glob("*.log") if _is_archive_log(p)), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _segment_meta_path(log_dir: Path) -> Path:
    return log_dir / SEGMENT_META_NAME


def _read_segment_start(log_dir: Path) -> float | None:
    meta = _segment_meta_path(log_dir)
    try:
        return float(meta.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _write_segment_start(log_dir: Path, ts: float | None = None) -> float:
    ts = ts if ts is not None else datetime.now().timestamp()
    try:
        _segment_meta_path(log_dir).write_text(str(ts), encoding="utf-8")
    except OSError:
        pass
    return ts


def _clear_segment_start(log_dir: Path) -> None:
    try:
        _segment_meta_path(log_dir).unlink(missing_ok=True)
    except OSError:
        pass


def _get_segment_start(log_dir: Path) -> float:
    """读取当前 rag.log 分段开始时间；缺失时回退到 rag.log mtime 并补写元数据。"""
    runtime_log = log_dir / RUNTIME_LOG_NAME
    saved = _read_segment_start(log_dir)
    if saved is not None:
        return saved
    if runtime_log.exists():
        try:
            return _write_segment_start(log_dir, runtime_log.stat().st_mtime)
        except OSError:
            pass
    return _write_segment_start(log_dir)


def _next_archive_path(log_dir: Path) -> Path:
    """生成不冲突的归档文件名：YYYYmmdd_HHMMSS(.n).log。"""
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = log_dir / f"{base}.log"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = log_dir / f"{base}.{idx}.log"
        if not candidate.exists():
            return candidate
        idx += 1


def _rollover_runtime_log_if_needed(log_dir: Path) -> Path:
    """
    检查 rag.log 是否超过轮转阈值：
    - 超过 5 小时：归档为时间戳日志并新建 rag.log。
    - 未超过：继续追加写入当前 rag.log。
    """
    runtime_log = log_dir / RUNTIME_LOG_NAME
    if not runtime_log.exists():
        return runtime_log

    age_seconds = datetime.now().timestamp() - _get_segment_start(log_dir)
    if age_seconds < ROTATE_HOURS * 3600:
        return runtime_log

    archive = _next_archive_path(log_dir)
    try:
        runtime_log.rename(archive)
        _clear_segment_start(log_dir)
    except OSError:
        # 归档失败时保底不丢日志：继续往原文件追加。
        return runtime_log
    return runtime_log


class RuntimeRotatingFileHandler(logging.FileHandler):
    """固定写 rag.log，并在持续运行超过指定秒数后归档为时间戳文件。"""

    def __init__(self, log_dir: Path, interval_seconds: int, encoding: str = "utf-8"):
        self.log_dir = log_dir
        self.interval_seconds = interval_seconds
        self.runtime_log = log_dir / RUNTIME_LOG_NAME
        self.segment_start_ts = _get_segment_start(log_dir)
        super().__init__(self.runtime_log, mode="a", encoding=encoding)

    def _rollover(self) -> None:
        self.acquire()
        try:
            if self.stream:
                self.stream.flush()
                self.stream.close()
                self.stream = None

            archive = _next_archive_path(self.log_dir)
            try:
                self.runtime_log.rename(archive)
            except OSError:
                # 重命名失败时重开 rag.log，避免阻塞日志写入。
                pass

            self.stream = self._open()
            self.segment_start_ts = _write_segment_start(self.log_dir)
            _cleanup_old_logs(self.log_dir)
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        if datetime.now().timestamp() - self.segment_start_ts >= self.interval_seconds:
            self._rollover()
        super().emit(record)


def setup_logging(log_dir: str, logger_name: str = "rag") -> str:
    """
    为指定 logger 配置双输出；返回当前运行中的 rag.log 路径。
    重复调用会清空该 logger 已有 handler，避免重复打印。
    文件日志持续写入 rag.log，当当前分段超过 5 小时会归档后重建。
    仅保留最近 MAX_LOG_FILES 个归档日志（不含 rag.log）。
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    runtime_log = _rollover_runtime_log_if_needed(path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]: %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = RuntimeRotatingFileHandler(path, interval_seconds=ROTATE_HOURS * 3600, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    _cleanup_old_logs(path)
    return str(runtime_log)


def get_logger(name: str = "rag") -> logging.Logger:
    """按名称取 logger；若未先 setup_logging，行为取决于根 logger 配置。"""
    return logging.getLogger(name)
