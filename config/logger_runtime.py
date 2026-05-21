"""标准库 logging 封装：控制台 INFO + 按次运行的文件 DEBUG。"""
import logging
from datetime import datetime
from pathlib import Path

MAX_LOG_FILES = 50


def _cleanup_old_logs(log_dir: Path, keep: int = MAX_LOG_FILES) -> None:
    """仅保留最新的 keep 个 .log 文件，其余按修改时间倒序删除。"""
    try:
        files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass


def setup_logging(log_dir: str, logger_name: str = "rag") -> str:
    """
    为指定 logger 配置双输出；返回本次运行日志文件路径。
    重复调用会清空该 logger 已有 handler，避免重复打印。
    仅保留最近 MAX_LOG_FILES 个日志文件，更早的会被清理。
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_file = path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]: %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(run_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    _cleanup_old_logs(path)
    return str(run_file)


def get_logger(name: str = "rag") -> logging.Logger:
    """按名称取 logger；若未先 setup_logging，行为取决于根 logger 配置。"""
    return logging.getLogger(name)
