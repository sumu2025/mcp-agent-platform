"""
基础API客户端 - 定义统一的API调用接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ModelClient(ABC):
    """
    模型客户端抽象基类，定义所有模型客户端必须实现的接口
    """
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        生成文本

        Args:
            prompt: 提示词
            max_tokens: 生成的最大token数量
            temperature: 温度参数，控制随机性
            **kwargs: 其他模型特定参数

        Returns:
            生成的文本内容
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            包含模型名称、版本等信息的字典
        """
        pass


# 在后续的开发中，我们将实现具体的模型客户端类，如ClaudeClient和DeepSeekClient
