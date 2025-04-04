"""
统一API客户端 - 提供统一的API调用接口
"""

from typing import Dict, Optional, Type

from mcp.api.base import ModelClient
from mcp.api.deepseek import DeepSeekClient
from mcp.api.mock import MockClient
from mcp.utils.exceptions import ConfigurationError
from mcp.utils.logger import app_logger


class MCPClient:
    """
    MCP统一客户端，管理多个模型客户端并提供统一的调用接口
    """
    
    # 客户端类映射表
    _client_classes: Dict[str, Type[ModelClient]] = {
        "deepseek": DeepSeekClient,
        "mock": MockClient,
        # "claude": ClaudeClient,  # 稍后添加
    }
    
    def __init__(self, default_model: str = "deepseek"):
        """
        初始化MCP客户端
        
        Args:
            default_model: 默认使用的模型提供商，支持"deepseek"和"claude"（未实现）
        """
        if default_model not in self._client_classes:
            raise ConfigurationError(f"不支持的模型提供商: {default_model}，"
                                    f"可用选项: {', '.join(self._client_classes.keys())}")
        
        self.default_model = default_model
        self._clients: Dict[str, ModelClient] = {}
        
        app_logger.info(f"MCP客户端初始化完成，默认模型提供商: {default_model}")
    
    def get_client(self, provider: Optional[str] = None) -> ModelClient:
        """
        获取指定提供商的客户端
        
        Args:
            provider: 模型提供商名称，如果为None则使用默认值
            
        Returns:
            ModelClient实例
            
        Raises:
            ConfigurationError: 如果提供商不支持
        """
        provider = provider or self.default_model
        
        if provider not in self._client_classes:
            raise ConfigurationError(f"不支持的模型提供商: {provider}，"
                                    f"可用选项: {', '.join(self._client_classes.keys())}")
        
        # 如果客户端不存在，则创建
        if provider not in self._clients:
            client_class = self._client_classes[provider]
            self._clients[provider] = client_class()
            app_logger.debug(f"创建了新的{provider}客户端")
        
        return self._clients[provider]
    
    def generate(self, prompt: str, provider: Optional[str] = None, **kwargs) -> str:
        """
        生成文本，自动使用指定的模型提供商
        
        Args:
            prompt: 提示词
            provider: 模型提供商，如果为None则使用默认值
            **kwargs: 传递给模型客户端的其他参数
            
        Returns:
            生成的文本
        """
        client = self.get_client(provider)
        return client.generate(prompt, **kwargs)
