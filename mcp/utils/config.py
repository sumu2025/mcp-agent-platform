"""
配置管理模块 - 加载和管理环境变量
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


class Config:
    """
    配置管理类，负责加载和访问环境变量
    """

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, env_file: Optional[str] = None):
        """单例模式实现，确保全局只有一个配置实例"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(env_file)
        return cls._instance

    def _load_config(self, env_file: Optional[str] = None) -> None:
        """
        加载配置文件和环境变量

        Args:
            env_file: .env文件的路径，如果为None则尝试在当前目录查找
        """
        # 尝试加载.env文件
        if env_file:
            load_dotenv(env_file)
        else:
            # 查找可能的.env文件位置
            possible_env_files = [
                ".env",
                Path(__file__).parent.parent.parent / ".env",
            ]
            
            for file_path in possible_env_files:
                if os.path.exists(file_path):
                    load_dotenv(file_path)
                    break

        # 加载关键配置
        self._config = {
            # Claude API配置
            "claude_api_key": os.getenv("CLAUDE_API_KEY", ""),
            "claude_api_url": os.getenv(
                "CLAUDE_API_URL", "https://api.anthropic.com/v1/messages"
            ),
            
            # DeepSeek API配置
            "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "deepseek_api_url": os.getenv(
                "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"
            ),
            
            # 缓存配置
            "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
            "cache_dir": os.getenv("CACHE_DIR", "./cache"),
            "cache_expiry": int(os.getenv("CACHE_EXPIRY", "86400")),
            
            # 日志配置
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_file": os.getenv("LOG_FILE", "./mcp.log"),
            
            # Obsidian配置
            "obsidian_vault_path": os.getenv("OBSIDIAN_VAULT_PATH", ""),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键名
            default: 如果键不存在，返回的默认值

        Returns:
            配置值或默认值
        """
        return self._config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """
        通过字典访问语法获取配置值

        Args:
            key: 配置键名

        Returns:
            配置值

        Raises:
            KeyError: 如果键不存在
        """
        if key not in self._config:
            raise KeyError(f"Configuration key '{key}' not found")
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """
        检查配置键是否存在

        Args:
            key: 配置键名

        Returns:
            如果键存在则返回True，否则返回False
        """
        return key in self._config
