"""标准库 logging 封装：控制台 INFO + 按次运行的文件 DEBUG。"""
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str, logger_name: str = "rag") -> str:
    """
    为指定 logger 配置双输出；返回本次运行日志文件路径。
    重复调用会清空该 logger 已有 handler，避免重复打印。
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
    return str(run_file)


def get_logger(name: str = "rag") -> logging.Logger:
    """按名称取 logger；若未先 setup_logging，行为取决于根 logger 配置。"""
    return logging.getLogger(name)
