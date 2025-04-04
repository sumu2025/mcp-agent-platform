"""
缓存基础模块 - 定义缓存的抽象接口和基本数据结构
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from mcp.utils.logger import app_logger


class CacheItem:
    """缓存项，包含值和元数据"""
    
    def __init__(
        self, 
        key: str, 
        value: Any, 
        expire_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
            expire_time: 过期时间戳（秒），如果为None则永不过期
            metadata: 附加元数据
        """
        self.key = key
        self.value = value
        self.expire_time = expire_time
        self.metadata = metadata or {}
        self.created_time = time.time()
        self.last_access_time = self.created_time
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """
        检查缓存项是否已过期
        
        Returns:
            如果已过期返回True，否则返回False
        """
        if self.expire_time is None:
            return False
        return time.time() > self.expire_time
    
    def access(self) -> None:
        """记录访问，更新访问时间和计数"""
        self.last_access_time = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """
        获取缓存项的年龄（秒）
        
        Returns:
            从创建到现在的秒数
        """
        return time.time() - self.created_time
    
    def get_time_to_live(self) -> Optional[float]:
        """
        获取剩余生存时间（秒）
        
        Returns:
            剩余生存时间，如果永不过期则返回None
        """
        if self.expire_time is None:
            return None
        ttl = self.expire_time - time.time()
        return max(0.0, ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将缓存项转换为字典
        
        Returns:
            包含缓存项所有信息的字典
        """
        return {
            "key": self.key,
            "value": self.value,
            "created_time": self.created_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "expire_time": self.expire_time,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheItem":
        """
        从字典创建缓存项
        
        Args:
            data: 包含缓存项信息的字典
            
        Returns:
            创建的缓存项
        """
        item = cls(
            key=data["key"],
            value=data["value"],
            expire_time=data.get("expire_time"),
            metadata=data.get("metadata", {})
        )
        item.created_time = data.get("created_time", item.created_time)
        item.last_access_time = data.get("last_access_time", item.last_access_time)
        item.access_count = data.get("access_count", 0)
        return item


class BaseCache(ABC):
    """缓存抽象基类，定义缓存操作接口"""
    
    def __init__(self, name: str = "default"):
        """
        初始化缓存
        
        Args:
            name: 缓存名称
        """
        self.name = name
        self.hit_count = 0
        self.miss_count = 0
        self.app_logger = app_logger
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期则返回None
        """
        pass
    
    @abstractmethod
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），如果为None则永不过期
            metadata: 附加元数据
            
        Returns:
            成功返回True，失败返回False
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            存在并删除成功返回True，不存在返回False
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        检查缓存键是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            如果存在且未过期返回True，否则返回False
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        清空缓存
        
        Returns:
            成功返回True，失败返回False
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含统计信息的字典
        """
        pass
    
    @abstractmethod
    def get_keys(self) -> List[str]:
        """
        获取所有缓存键
        
        Returns:
            缓存键列表
        """
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """
        获取缓存项数量
        
        Returns:
            缓存项数量
        """
        pass
    
    def get_hit_ratio(self) -> float:
        """
        获取缓存命中率
        
        Returns:
            命中率（0-1之间的浮点数）
        """
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total
    
    def record_hit(self) -> None:
        """记录缓存命中"""
        self.hit_count += 1
    
    def record_miss(self) -> None:
        """记录缓存未命中"""
        self.miss_count += 1


def generate_cache_key(prefix: str, args: Any, kwargs: Any) -> str:
    """
    生成缓存键
    
    Args:
        prefix: 前缀
        args: 位置参数
        kwargs: 关键字参数
        
    Returns:
        生成的缓存键
    """
    # 确保args和kwargs可以被序列化
    if isinstance(args, tuple):
        serializable_args = [arg for arg in args if is_serializable(arg)]
    else:
        serializable_args = args if is_serializable(args) else str(args)
    
    if isinstance(kwargs, dict):
        serializable_kwargs = {
            k: v for k, v in kwargs.items() 
            if k != "self" and is_serializable(v)
        }
    else:
        serializable_kwargs = kwargs if is_serializable(kwargs) else str(kwargs)
    
    # 构建用于哈希的字符串
    key_str = f"{prefix}:{serializable_args}:{serializable_kwargs}"
    
    # 计算MD5哈希
    return hashlib.md5(key_str.encode()).hexdigest()


def is_serializable(obj: Any) -> bool:
    """
    检查对象是否可序列化为JSON
    
    Args:
        obj: 要检查的对象
        
    Returns:
        如果可序列化返回True，否则返回False
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def cache_decorator(cache_instance: BaseCache, ttl: Optional[int] = None):
    """
    缓存装饰器，用于缓存函数调用结果
    
    Args:
        cache_instance: 缓存实例
        ttl: 缓存生存时间（秒）
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取函数的全名作为前缀
            prefix = f"{func.__module__}.{func.__name__}"
            
            # 生成缓存键
            cache_key = generate_cache_key(prefix, args, kwargs)
            
            # 尝试从缓存获取结果
            result = cache_instance.get(cache_key)
            if result is not None:
                app_logger.debug(f"缓存命中: {cache_key}")
                return result
            
            # 缓存未命中，调用原函数
            app_logger.debug(f"缓存未命中: {cache_key}")
            result = func(*args, **kwargs)
            
            # 将结果存入缓存
            if is_serializable(result):
                metadata = {
                    "function": prefix,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "timestamp": datetime.now().isoformat()
                }
                cache_instance.set(cache_key, result, ttl, metadata)
                app_logger.debug(f"结果已缓存: {cache_key}")
            else:
                app_logger.warning(f"结果不可序列化，未缓存: {cache_key}")
            
            return result
        return wrapper
    return decorator
