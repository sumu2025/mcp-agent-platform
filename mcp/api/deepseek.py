"""
DeepSeek API客户端 - 实现对DeepSeek API的调用
"""

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel, Field

from mcp.api.base import ModelClient
from mcp.utils.cache import cache_manager
from mcp.utils.config import Config
from mcp.utils.exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    InvalidRequestError,
    ModelError,
    RateLimitError,
)
from mcp.utils.logger import app_logger


class DeepSeekMessage(BaseModel):
    """DeepSeek API消息模型"""
    role: str = Field(..., description="消息角色，如system, user, assistant")
    content: str = Field(..., description="消息内容")


class DeepSeekRequest(BaseModel):
    """DeepSeek API请求模型"""
    model: str = Field(..., description="要使用的模型名称")
    messages: List[DeepSeekMessage] = Field(..., description="消息历史")
    temperature: float = Field(0.7, description="生成的随机性")
    max_tokens: Optional[int] = Field(None, description="生成的最大token数量")
    stream: bool = Field(False, description="是否流式输出")


class DeepSeekClient(ModelClient):
    """
    DeepSeek API客户端，用于调用DeepSeek模型API
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "deepseek-chat",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: int = 2,
    ):
        """
        初始化DeepSeek API客户端

        Args:
            api_key: DeepSeek API密钥，如果为None则从配置中读取
            api_url: DeepSeek API地址，如果为None则从配置中读取
            model: 默认使用的模型
            timeout: API调用超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        # 获取配置
        config = Config()
        self.api_key = api_key or config.get("deepseek_api_key")
        self.api_url = api_url or config.get("deepseek_api_url")
        
        # 检查API密钥和URL
        if not self.api_key:
            raise AuthenticationError("DeepSeek API密钥未设置，请在.env文件中设置DEEPSEEK_API_KEY")
        
        if not self.api_url:
            # 使用默认的API URL
            self.api_url = "https://api.deepseek.com/v1/chat/completions"
            app_logger.warning(f"DeepSeek API URL未设置，使用默认值: {self.api_url}")
        
        # 设置客户端参数
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 设置API请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # 缓存相关设置
        self.use_cache = config.get("cache_enabled", True)
        self.cache_ttl = int(config.get("cache_expiry", 86400))  # 默认缓存1天
        
        app_logger.info(f"DeepSeek API客户端初始化完成，使用模型: {self.model}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        生成文本

        Args:
            prompt: 用户提示词
            max_tokens: 生成的最大token数量
            temperature: 温度参数，控制随机性
            system_prompt: 系统提示词，设置模型行为
            **kwargs: 其他模型特定参数

        Returns:
            生成的文本内容

        Raises:
            APIError: API调用发生错误
        """
        # 构建消息列表
        messages = []
        
        # 添加系统提示（如果有）
        if system_prompt:
            messages.append(DeepSeekMessage(role="system", content=system_prompt))
        
        # 添加用户提示
        messages.append(DeepSeekMessage(role="user", content=prompt))
        
        # 构建请求体
        request_data = DeepSeekRequest(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        
        # 检查缓存
        if self.use_cache:
            cache_key = self._generate_cache_key(request_data)
            cached_response = cache_manager.get(cache_key)
            if cached_response is not None:
                app_logger.debug(f"使用缓存的响应: {cache_key}")
                return cached_response
        
        # 发起API请求
        app_logger.debug(f"请求DeepSeek API: {json.dumps(request_data.model_dump())}")
        
        try:
            response = self._make_request(request_data)
            
            # 缓存响应
            if self.use_cache:
                cache_metadata = {
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timestamp": time.time()
                }
                cache_manager.set(cache_key, response, self.cache_ttl, cache_metadata)
                app_logger.debug(f"响应已缓存: {cache_key}")
            
            return response
        except Exception as e:
            app_logger.error(f"DeepSeek API调用失败: {str(e)}")
            raise

    def _make_request(self, request_data: DeepSeekRequest) -> str:
        """
        发起API请求，包含重试逻辑

        Args:
            request_data: 请求数据

        Returns:
            生成的文本内容

        Raises:
            APIError: API调用发生错误
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=request_data.model_dump(),
                    timeout=self.timeout,
                )
                
                # 检查响应状态
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # 提取生成的文本
                    if "choices" in response_data and len(response_data["choices"]) > 0:
                        message = response_data["choices"][0].get("message", {})
                        content = message.get("content", "")
                        return content
                    else:
                        raise ModelError("DeepSeek API返回的数据格式异常，找不到生成的内容")
                
                # 处理常见错误
                elif response.status_code == 401:
                    raise AuthenticationError("DeepSeek API密钥认证失败")
                elif response.status_code == 429:
                    raise RateLimitError("DeepSeek API请求过于频繁，请稍后再试")
                elif response.status_code == 400:
                    raise InvalidRequestError(f"DeepSeek API请求参数错误: {response.text}")
                else:
                    raise APIError(
                        f"DeepSeek API请求失败，状态码: {response.status_code}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )
                    
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = APIConnectionError(f"DeepSeek API连接错误: {str(e)}")
            except requests.Timeout:
                last_error = APITimeoutError(f"DeepSeek API请求超时，超时设置为 {self.timeout}秒")
            except (ModelError, AuthenticationError, RateLimitError, InvalidRequestError, APIError) as e:
                # 这些错误不需要重试
                raise e
            except Exception as e:
                last_error = APIError(f"调用DeepSeek API时发生未知错误: {str(e)}")
            
            # 增加重试计数并等待
            retries += 1
            if retries <= self.max_retries:
                wait_time = self.retry_delay * (2 ** (retries - 1))  # 指数退避
                app_logger.warning(f"DeepSeek API请求失败，{wait_time}秒后重试 ({retries}/{self.max_retries})")
                time.sleep(wait_time)
        
        # 所有重试都失败了
        if last_error:
            raise last_error
        else:
            raise APIError("DeepSeek API请求失败，已达到最大重试次数")
    
    def _generate_cache_key(self, request_data: DeepSeekRequest) -> str:
        """
        生成缓存键
        
        Args:
            request_data: 请求数据
            
        Returns:
            缓存键
        """
        # 序列化请求数据，确保包含所有影响结果的参数
        key_data = {
            "model": request_data.model,
            "messages": [m.model_dump() for m in request_data.messages],
            "temperature": request_data.temperature,
            "max_tokens": request_data.max_tokens,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        
        # 使用MD5哈希作为缓存键
        return f"deepseek:{hashlib.md5(key_str.encode()).hexdigest()}"

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            包含模型名称、版本等信息的字典
        """
        return {
            "provider": "DeepSeek",
            "model": self.model,
            "api_version": "v1",
        }
