"""
日志模块 - 配置和管理项目日志
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .config import Config


def setup_logger(
    name: str, 
    log_level: Optional[str] = None, 
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    设置并配置日志记录器

    Args:
        name: 日志记录器名称
        log_level: 日志级别，如果为None则从配置中读取
        log_file: 日志文件路径，如果为None则从配置中读取

    Returns:
        配置好的Logger实例
    """
    # 加载配置
    config = Config()
    
    # 设置日志级别
    if log_level is None:
        log_level = config.get("log_level", "INFO")
    
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = log_level_map.get(log_level.upper(), logging.INFO)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器（如果配置了日志文件）
    if log_file is None:
        log_file = config.get("log_file")
        
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 创建默认的应用日志记录器
app_logger = setup_logger("mcp")
