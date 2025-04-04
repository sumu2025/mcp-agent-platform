"""
缓存管理模块 - 集成和管理各种缓存实现
"""

import os
import time
from typing import Any, Dict, List, Optional, Type, Union

from mcp.utils.cache_base import BaseCache, CacheItem, cache_decorator
from mcp.utils.cache_disk import DiskCache
from mcp.utils.cache_memory import MemoryCache
from mcp.utils.config import Config
from mcp.utils.exceptions import CacheError
from mcp.utils.logger import app_logger


class CacheManager:
    """
    缓存管理器，管理多个缓存实例
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        enabled: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        default_ttl: Optional[int] = None,
        memory_max_size: Optional[int] = None,
        disk_max_size: Optional[int] = None,
    ):
        """
        初始化缓存管理器
        
        Args:
            enabled: 是否启用缓存，如果为None则从配置获取
            cache_dir: 缓存目录路径，如果为None则从配置获取
            default_ttl: 默认缓存生存时间（秒），如果为None则从配置获取
            memory_max_size: 内存缓存最大项数，如果为None则从配置获取
            disk_max_size: 磁盘缓存最大项数，如果为None则从配置获取
        """
        # 避免重复初始化
        if self._initialized:
            return
        
        # 加载配置
        config = Config()
        self.enabled = enabled if enabled is not None else config.get("cache_enabled", True)
        self.cache_dir = cache_dir or config.get("cache_dir", "./cache")
        self.default_ttl = default_ttl or int(config.get("cache_expiry", 86400))
        
        # 确保缓存目录存在
        if self.enabled and not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
            except Exception as e:
                app_logger.error(f"创建缓存目录失败: {str(e)}")
                self.enabled = False
        
        # 创建缓存实例
        self._caches: Dict[str, BaseCache] = {}
        
        if self.enabled:
            # 创建内存缓存
            memory_size = memory_max_size or int(config.get("memory_cache_max_size", 1000))
            self._memory_cache = MemoryCache(
                name="memory", 
                max_size=memory_size,
                cleanup_interval=300
            )
            
            # 创建磁盘缓存
            disk_size = disk_max_size or int(config.get("disk_cache_max_size", 10000))
            self._disk_cache = DiskCache(
                name="disk",
                cache_dir=self.cache_dir,
                max_size=disk_size,
                cleanup_interval=3600
            )
            
            # 注册默认缓存
            self.register_cache("memory", self._memory_cache)
            self.register_cache("disk", self._disk_cache)
            
            app_logger.info(f"缓存系统初始化完成，启用状态: {self.enabled}")
        else:
            app_logger.info("缓存系统已禁用")
        
        self._initialized = True
    
    def register_cache(self, name: str, cache: BaseCache) -> None:
        """
        注册缓存实例
        
        Args:
            name: 缓存名称
            cache: 缓存实例
        """
        if not self.enabled:
            return
        
        self._caches[name] = cache
        app_logger.debug(f"注册缓存: {name}, 类型: {type(cache).__name__}")
    
    def get_cache(self, name: str) -> Optional[BaseCache]:
        """
        获取缓存实例
        
        Args:
            name: 缓存名称
            
        Returns:
            缓存实例，如果不存在则返回None
        """
        if not self.enabled:
            return None
        
        return self._caches.get(name)
    
    def get(
        self, 
        key: str, 
        cache_name: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """
        从缓存获取值
        
        Args:
            key: 缓存键
            cache_name: 缓存名称，如果为None则尝试从内存缓存和磁盘缓存获取
            default: 缓存不存在时返回的默认值
            
        Returns:
            缓存值，如果不存在则返回默认值
        """
        if not self.enabled:
            return default
        
        # 如果指定了缓存名称，则直接从该缓存获取
        if cache_name:
            cache = self.get_cache(cache_name)
            if cache:
                value = cache.get(key)
                return value if value is not None else default
            return default
        
        # 否则，先从内存缓存获取，再从磁盘缓存获取
        value = self._memory_cache.get(key)
        if value is not None:
            return value
        
        # 从磁盘缓存获取
        value = self._disk_cache.get(key)
        if value is not None:
            # 获取成功后，将值存入内存缓存
            self._memory_cache.set(key, value)
            return value
        
        return default
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cache_names: Optional[List[str]] = None
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），如果为None则使用默认值
            metadata: 附加元数据
            cache_names: 要设置的缓存名称列表，如果为None则设置内存缓存和磁盘缓存
            
        Returns:
            成功返回True，失败返回False
        """
        if not self.enabled:
            return False
        
        # 使用默认TTL
        if ttl is None:
            ttl = self.default_ttl
        
        # 如果指定了缓存名称，则设置指定的缓存
        if cache_names:
            success = True
            for name in cache_names:
                cache = self.get_cache(name)
                if cache:
                    if not cache.set(key, value, ttl, metadata):
                        success = False
            return success
        
        # 否则，同时设置内存缓存和磁盘缓存
        memory_success = self._memory_cache.set(key, value, ttl, metadata)
        disk_success = self._disk_cache.set(key, value, ttl, metadata)
        
        return memory_success and disk_success
    
    def delete(self, key: str, cache_names: Optional[List[str]] = None) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            cache_names: 要删除的缓存名称列表，如果为None则删除所有缓存
            
        Returns:
            存在并删除成功返回True，不存在返回False
        """
        if not self.enabled:
            return False
        
        # 如果指定了缓存名称，则删除指定的缓存
        if cache_names:
            success = False
            for name in cache_names:
                cache = self.get_cache(name)
                if cache:
                    if cache.delete(key):
                        success = True
            return success
        
        # 否则，删除所有缓存
        success = False
        for cache in self._caches.values():
            if cache.delete(key):
                success = True
        
        return success
    
    def exists(self, key: str, cache_name: Optional[str] = None) -> bool:
        """
        检查缓存键是否存在且未过期
        
        Args:
            key: 缓存键
            cache_name: 缓存名称，如果为None则检查任意缓存
            
        Returns:
            如果存在且未过期返回True，否则返回False
        """
        if not self.enabled:
            return False
        
        # 如果指定了缓存名称，则检查指定的缓存
        if cache_name:
            cache = self.get_cache(cache_name)
            return cache.exists(key) if cache else False
        
        # 否则，检查所有缓存
        return any(cache.exists(key) for cache in self._caches.values())
    
    def clear(self, cache_names: Optional[List[str]] = None) -> bool:
        """
        清空缓存
        
        Args:
            cache_names: 要清空的缓存名称列表，如果为None则清空所有缓存
            
        Returns:
            成功返回True，失败返回False
        """
        if not self.enabled:
            return False
        
        # 如果指定了缓存名称，则清空指定的缓存
        if cache_names:
            success = True
            for name in cache_names:
                cache = self.get_cache(name)
                if cache:
                    if not cache.clear():
                        success = False
            return success
        
        # 否则，清空所有缓存
        success = True
        for cache in self._caches.values():
            if not cache.clear():
                success = False
        
        return success
    
    def get_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Args:
            cache_name: 缓存名称，如果为None则获取所有缓存的统计信息
            
        Returns:
            包含统计信息的字典
        """
        if not self.enabled:
            return {"enabled": False}
        
        # 如果指定了缓存名称，则获取指定的缓存统计信息
        if cache_name:
            cache = self.get_cache(cache_name)
            return cache.get_stats() if cache else {}
        
        # 否则，获取所有缓存的统计信息
        stats = {
            "enabled": self.enabled,
            "cache_dir": self.cache_dir,
            "default_ttl": self.default_ttl,
            "caches": {name: cache.get_stats() for name, cache in self._caches.items()}
        }
        
        # 计算总体命中率
        total_hits = sum(cache.hit_count for cache in self._caches.values())
        total_misses = sum(cache.miss_count for cache in self._caches.values())
        total = total_hits + total_misses
        
        if total > 0:
            stats["overall_hit_ratio"] = total_hits / total
        else:
            stats["overall_hit_ratio"] = 0
        
        return stats
    
    def cache_function(
        self, 
        ttl: Optional[int] = None,
        cache_name: str = "memory"
    ):
        """
        函数缓存装饰器
        
        Args:
            ttl: 缓存生存时间（秒），如果为None则使用默认值
            cache_name: 要使用的缓存名称
            
        Returns:
            装饰器函数
        """
        if not self.enabled:
            return lambda func: func
        
        cache = self.get_cache(cache_name)
        if not cache:
            return lambda func: func
        
        return cache_decorator(cache, ttl or self.default_ttl)


# 创建全局缓存管理器实例
cache_manager = CacheManager()
