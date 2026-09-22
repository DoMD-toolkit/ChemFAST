from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import IO, Iterator

LOGGER_NAME = "task_logger"

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(name)s " "(%(filename)s:%(lineno)d): %(message)s"

LOGGER_FORMATTER = logging.Formatter(LOGGER_FORMAT)

# 用于识别本模块创建的控制台 Handler。
_CONSOLE_HANDLER_MARK = "_chemfast_console_handler"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """
    获取指定名称的 logger。

    同一进程内，相同 name 始终对应同一个 Logger 对象。
    """
    return logging.getLogger(name)


# 这是供内部模块统一 import 的文件/控制台 logger：
#
#     from chemfast.misc.logger import logger
#
# 它与 redis_wkr 创建并传入 handler(...) 的 SSE logger 完全独立。
logger = get_logger()
logger.setLevel(logging.INFO)
logger.propagate = False


def _find_console_handler() -> logging.StreamHandler | None:
    """查找由本模块创建的控制台 Handler。"""
    for handler in logger.handlers:
        if getattr(handler, _CONSOLE_HANDLER_MARK, False):
            return handler
    return None


def enable_console_logging(level: int = logging.INFO, stream: IO[str] | None = None) -> logging.StreamHandler:
    """
    开启全局 logger 的控制台输出。

    重复调用不会添加重复 Handler。
    """
    if stream is None:
        stream = sys.stdout

    handler = _find_console_handler()
    if handler is not None:
        handler.setLevel(level)
        if handler.stream is not stream:
            handler.setStream(stream)
        return handler

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(LOGGER_FORMATTER)
    setattr(handler, _CONSOLE_HANDLER_MARK, True)

    logger.addHandler(handler)
    return handler


def disable_console_logging() -> None:
    """
    永久关闭当前进程中全局 logger 的控制台输出。

    不影响任何 FileHandler。
    """
    handler = _find_console_handler()
    if handler is None:
        return

    logger.removeHandler(handler)
    handler.close()


def console_logging_enabled() -> bool:
    """返回全局 logger 当前是否输出到控制台。"""
    return _find_console_handler() is not None


def _create_file_handler(log_path: str, level: int = logging.INFO) -> logging.FileHandler:
    """创建配置完成的 UTF-8 文件 Handler。"""
    handler = logging.FileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(LOGGER_FORMATTER)
    return handler


@contextmanager
def task_file_log_scope(task_name: str, log_dir: str, level: int = logging.INFO) -> Iterator[str]:
    """
    建立任务级文件日志作用域。

    进入作用域时：
        1. 临时摘掉本模块的控制台 Handler；
        2. 添加任务 FileHandler。

    退出作用域时：
        1. 移除并关闭任务 FileHandler；
        2. 恢复进入作用域之前的控制台 Handler。

    因此：
        - 普通代码默认输出到控制台；
        - Redis 工作函数进入该 scope 后，全局 logger 自动只写文件；
        - 不需要在 redis_wkr 中额外调用 disable_console_logging()；
        - 不影响 redis_wkr 单独创建并传入的 SSE logger。

    作用域内所有通过以下共享 logger 产生的日志：

        from chemfast.misc.logger import logger

    都会写入：

        <log_dir>/<task_name>_debug.log

    如果内部再进入 mol_file_log_scope()，分子日志会同时写入：
        1. 当前任务总日志；
        2. 当前分子日志。
    """
    os.makedirs(log_dir, exist_ok=True)

    debug_log_path = os.path.join(
        log_dir,
        f"{task_name}_debug.log",
    )

    file_handler = _create_file_handler(
        debug_log_path,
        level=level,
    )

    # 只临时摘掉本模块自己创建的控制台 Handler。
    # 不关闭它，因为退出 scope 后还要原样恢复。
    console_handler = _find_console_handler()
    if console_handler is not None:
        logger.removeHandler(console_handler)

    logger.addHandler(file_handler)

    try:
        yield debug_log_path
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()

        if console_handler is not None:
            logger.addHandler(console_handler)


@contextmanager
def mol_file_log_scope(idx: int, log_dir: str, level: int = logging.INFO) -> Iterator[str]:
    """
    建立单个分子的文件日志作用域。

    这个 scope 与控制台控制完全独立：
        - 它不会关闭控制台；
        - 它不会开启控制台；
        - 它只临时增加一个分子 FileHandler。

    因此它既可以独立使用：

        with mol_file_log_scope(0, "./logs"):
            run_molecule()

    此时默认会同时写控制台和 MOL_000000.log。

    也可以嵌套在 task_file_log_scope() 中：

        with task_file_log_scope("task_001", results_dir):
            for idx, mol in enumerate(mols):
                with mol_file_log_scope(idx, results_dir):
                    run_molecule(mol)

    此时 task scope 已经临时关闭控制台，所以分子处理期间会写入：
        1. task_001_debug.log；
        2. MOL_<idx>.log；
        3. 不输出到控制台。
    """
    os.makedirs(log_dir, exist_ok=True)

    mol_log_path = os.path.join(
        log_dir,
        f"MOL_{idx:06d}.log",
    )

    file_handler = _create_file_handler(
        mol_log_path,
        level=level,
    )

    logger.addHandler(file_handler)

    try:
        yield mol_log_path
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


# 普通单独执行是默认场景，因此模块导入后默认启用控制台。
# task_file_log_scope() 会在自己的作用域内自动临时关闭它。
enable_console_logging()
