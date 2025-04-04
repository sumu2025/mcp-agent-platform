"""
模拟API客户端 - 用于开发和测试时不需要实际API调用
"""

import json
import hashlib
import time
from typing import Any, Dict, Optional

from mcp.api.base import ModelClient
from mcp.utils.cache import cache_manager
from mcp.utils.config import Config
from mcp.utils.logger import app_logger


class MockClient(ModelClient):
    """
    模拟API客户端，用于开发和测试时不需要实际API调用
    """

    def __init__(
        self,
        model: str = "mock-model",
        response_time: float = 1.0,
        **kwargs
    ):
        """
        初始化模拟客户端

        Args:
            model: 模拟的模型名称
            response_time: 模拟的响应时间（秒）
        """
        # 获取配置
        config = Config()
        self.use_cache = config.get("cache_enabled", True)
        self.cache_ttl = int(config.get("cache_expiry", 86400))  # 默认缓存1天
        
        self.model = model
        self.response_time = response_time
        app_logger.info(f"模拟API客户端初始化完成，使用模型: {self.model}")

    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        生成文本（模拟）

        Args:
            prompt: 提示词
            max_tokens: 生成的最大token数量
            temperature: 温度参数，控制随机性
            **kwargs: 其他模型特定参数

        Returns:
            生成的文本内容
        """
        app_logger.info(f"模拟API调用: prompt={prompt[:30]}..., max_tokens={max_tokens}, temperature={temperature}")
        
        # 生成缓存键
        if self.use_cache:
            cache_key = self._generate_cache_key(prompt, max_tokens, temperature, kwargs)
            cached_response = cache_manager.get(cache_key)
            if cached_response is not None:
                app_logger.debug(f"使用缓存的模拟响应: {cache_key}")
                return cached_response
        
        # 模拟API调用延迟
        time.sleep(self.response_time)
        
        # 根据提示词生成简单的模拟回复
        response = self._generate_response(prompt)
        
        # 缓存响应
        if self.use_cache:
            cache_metadata = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timestamp": time.time(),
                "is_mock": True
            }
            cache_manager.set(cache_key, response, self.cache_ttl, cache_metadata)
            app_logger.debug(f"模拟响应已缓存: {cache_key}")
        
        return response
    
    def _generate_response(self, prompt: str) -> str:
        """根据提示词生成模拟响应"""
        if "python" in prompt.lower():
            return """Python是一种高级编程语言，具有以下特点和优势：

1. 简洁易读：Python的语法设计简洁明了，使用缩进表示代码块，代码可读性强。
2. 多用途：可用于Web开发、数据分析、人工智能、科学计算等多个领域。
3. 丰富的库和框架：拥有大量的第三方库和框架，如NumPy、Pandas、Django等。
4. 跨平台：Python可在Windows、MacOS、Linux等多种操作系统上运行。
5. 解释型语言：无需编译，直接运行代码，开发效率高。
6. 支持多种编程范式：包括面向对象、命令式和函数式编程。

Python特别适合初学者学习，同时也被专业开发者广泛使用于生产环境。"""
        
        elif "人工智能" in prompt.lower() or "机器学习" in prompt.lower():
            return """人工智能(AI)和机器学习(ML)的主要区别：

人工智能是一个更广泛的概念，指创建能够模拟人类智能行为的系统或机器。它是一个包罗万象的领域，目标是开发能思考、学习和解决问题的智能系统。

机器学习是人工智能的一个子集，专注于通过数据和经验自动学习和改进的算法和技术。ML系统能从数据中学习模式并做出决策，而无需明确编程每一步。

简单来说，所有的机器学习都是人工智能，但不是所有的人工智能都是机器学习。AI是目标（创建智能），而ML是实现这一目标的方法之一。"""
        
        else:
            return f"""这是一个模拟回复，用于开发和测试。

您的提问是："{prompt}"

在实际部署时，这将被替换为来自真实AI模型的回复。模拟模式主要用于开发过程中，以避免消耗API配额。

参数信息：
- 最大token数：{max_tokens}
- 温度：{temperature}
- 模型：{self.model}"""

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            包含模型名称、版本等信息的字典
        """
        return {
            "provider": "Mock",
            "model": self.model,
            "api_version": "mock-v1",
            "is_mock": True
        }
    
    def _generate_cache_key(self, prompt: str, max_tokens: int, temperature: float, kwargs: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 组合关键参数生成缓存键
        key_data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": kwargs.get("system_prompt", "")
        }
        
        # 序列化并哈希
        key_str = json.dumps(key_data, sort_keys=True)
        return f"mock:{hashlib.md5(key_str.encode()).hexdigest()}"
