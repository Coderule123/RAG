"""标准库 logging 封装：控制台 INFO + 文件日志（RAG 持续写入 rag.log 并轮转归档）。"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

MAX_LOG_FILES = 50
RUNTIME_LOG_NAME = "rag.log"
SEGMENT_META_NAME = "rag.log.segment"
ARCHIVE_SUBDIR_NAME = "rag"
ROTATE_HOURS = 5
LogMode = Literal["rotating", "timestamp"]


def _is_archive_log(path: Path, runtime_log_name: str = RUNTIME_LOG_NAME) -> bool:
    """是否为历史归档日志（时间戳文件），排除当前运行中的 rag.log。"""
    return path.is_file() and path.suffix == ".log" and path.name != runtime_log_name


def _cleanup_old_logs(log_dir: Path, keep: int = MAX_LOG_FILES, runtime_log_name: str = RUNTIME_LOG_NAME) -> None:
    """仅保留最新 keep 个归档日志，rag.log 不参与容量统计。"""
    try:
        files = sorted(
            (p for p in log_dir.glob("*.log") if _is_archive_log(p, runtime_log_name)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
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


def _next_timestamp_log_path(log_dir: Path) -> Path:
    """生成不冲突的时间戳日志名：YYYYmmdd_HHMMSS(.n).log。"""
    log_dir.mkdir(parents=True, exist_ok=True)
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


def _archive_dir(log_dir: Path) -> Path:
    return log_dir / ARCHIVE_SUBDIR_NAME


def _rollover_runtime_log_if_needed(log_dir: Path) -> Path:
    """
    检查 rag.log 是否超过轮转阈值：
    - 超过 5 小时：归档到 logs/rag/ 下时间戳日志并新建 rag.log。
    - 未超过：继续追加写入当前 rag.log。
    """
    runtime_log = log_dir / RUNTIME_LOG_NAME
    if not runtime_log.exists():
        return runtime_log

    age_seconds = datetime.now().timestamp() - _get_segment_start(log_dir)
    if age_seconds < ROTATE_HOURS * 3600:
        return runtime_log

    archive = _next_timestamp_log_path(_archive_dir(log_dir))
    try:
        runtime_log.rename(archive)
        _clear_segment_start(log_dir)
    except OSError:
        # 归档失败时保底不丢日志：继续往原文件追加。
        return runtime_log
    return runtime_log


class RuntimeRotatingFileHandler(logging.FileHandler):
    """固定写 rag.log，并在持续运行超过指定秒数后归档到 logs/rag/ 时间戳文件。"""

    def __init__(self, log_dir: Path, archive_dir: Path, interval_seconds: int, encoding: str = "utf-8"):
        self.log_dir = log_dir
        self.archive_dir = archive_dir
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

            archive = _next_timestamp_log_path(self.archive_dir)
            try:
                self.runtime_log.rename(archive)
            except OSError:
                # 重命名失败时重开 rag.log，避免阻塞日志写入。
                pass

            self.stream = self._open()
            self.segment_start_ts = _write_segment_start(self.log_dir)
            _cleanup_old_logs(self.archive_dir)
        finally:
            self.release()

    def emit(self, record: logging.LogRecord) -> None:
        if datetime.now().timestamp() - self.segment_start_ts >= self.interval_seconds:
            self._rollover()
        super().emit(record)


def _configure_logger(logger: logging.Logger) -> logging.Formatter:
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]: %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)
    return fmt


def _setup_rotating_logging(path: Path, logger_name: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    archive_path = _archive_dir(path)
    archive_path.mkdir(parents=True, exist_ok=True)
    runtime_log = _rollover_runtime_log_if_needed(path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = _configure_logger(logger)
    file_handler = RuntimeRotatingFileHandler(
        path, archive_path, interval_seconds=ROTATE_HOURS * 3600, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _cleanup_old_logs(archive_path)
    return str(runtime_log)


def _setup_timestamp_logging(path: Path, logger_name: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    log_file = _next_timestamp_log_path(path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = _configure_logger(logger)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    _cleanup_old_logs(path, runtime_log_name="")
    return str(log_file)


def setup_logging(
    log_dir: str,
    logger_name: str = "rag",
    *,
    log_mode: LogMode = "rotating",
) -> str:
    """
    为指定 logger 配置双输出；返回当前日志文件路径。
    重复调用会清空该 logger 已有 handler，避免重复打印。

    log_mode:
    - rotating（默认，RAG/rag_api.py）：持续写入 rag.log，超过 5 小时归档到 logs/rag/。
    - timestamp（DP/document_processor.py）：每次启动直接写入 logs/doc/YYYYmmdd_HHMMSS.log。
    """
    path = Path(log_dir)
    if log_mode == "timestamp":
        return _setup_timestamp_logging(path, logger_name)
    return _setup_rotating_logging(path, logger_name)


def get_logger(name: str = "rag") -> logging.Logger:
    """按名称取 logger；若未先 setup_logging，行为取决于根 logger 配置。"""
    return logging.getLogger(name)
