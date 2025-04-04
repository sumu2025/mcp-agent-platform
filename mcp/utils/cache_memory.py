"""
内存缓存模块 - 提供基于内存的缓存实现
"""

import time
from threading import RLock
from typing import Any, Dict, List, Optional, Set

from mcp.utils.cache_base import BaseCache, CacheItem
from mcp.utils.exceptions import CacheError


class MemoryCache(BaseCache):
    """基于内存的缓存实现"""
    
    def __init__(
        self, 
        name: str = "memory", 
        max_size: int = 1000,
        cleanup_interval: int = 60
    ):
        """
        初始化内存缓存
        
        Args:
            name: 缓存名称
            max_size: 最大缓存项数量
            cleanup_interval: 自动清理间隔（秒）
        """
        super().__init__(name)
        self._cache: Dict[str, CacheItem] = {}
        self._lock = RLock()
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.last_cleanup_time = time.time()
        self.app_logger.debug(f"创建内存缓存: {name}, 最大容量: {max_size}项")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期则返回None
        """
        with self._lock:
            item = self._cache.get(key)
            
            # 缓存项不存在
            if item is None:
                self.record_miss()
                return None
            
            # 缓存项已过期
            if item.is_expired():
                self.delete(key)
                self.record_miss()
                return None
            
            # 更新访问记录
            item.access()
            self.record_hit()
            
            # 检查是否需要清理
            self._auto_cleanup()
            
            return item.value
    
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
        with self._lock:
            # 计算过期时间
            expire_time = None
            if ttl is not None:
                expire_time = time.time() + ttl
            
            # 创建缓存项
            item = CacheItem(key, value, expire_time, metadata)
            
            # 检查缓存大小是否超过限制
            if len(self._cache) >= self.max_size and key not in self._cache:
                # 清理过期项
                self._cleanup_expired()
                
                # 如果仍然超过限制，则移除最不常用的项
                if len(self._cache) >= self.max_size:
                    self._evict_items(1)
            
            # 添加或更新缓存项
            self._cache[key] = item
            
            # 检查是否需要清理
            self._auto_cleanup()
            
            return True
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            存在并删除成功返回True，不存在返回False
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存键是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            如果存在且未过期返回True，否则返回False
        """
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                return False
            
            if item.is_expired():
                self.delete(key)
                return False
            
            return True
    
    def clear(self) -> bool:
        """
        清空缓存
        
        Returns:
            成功返回True，失败返回False
        """
        with self._lock:
            self._cache.clear()
            self.hit_count = 0
            self.miss_count = 0
            self.last_cleanup_time = time.time()
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含统计信息的字典
        """
        with self._lock:
            return {
                "name": self.name,
                "type": "memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_ratio": self.get_hit_ratio(),
                "expired_count": self._count_expired(),
                "last_cleanup_time": self.last_cleanup_time,
            }
    
    def get_keys(self) -> List[str]:
        """
        获取所有未过期的缓存键
        
        Returns:
            缓存键列表
        """
        with self._lock:
            keys = []
            for key, item in list(self._cache.items()):
                if not item.is_expired():
                    keys.append(key)
                else:
                    # 顺便删除过期项
                    self.delete(key)
            return keys
    
    def get_size(self) -> int:
        """
        获取缓存项数量（包括过期项）
        
        Returns:
            缓存项数量
        """
        with self._lock:
            return len(self._cache)
    
    def get_all_items(self) -> Dict[str, CacheItem]:
        """
        获取所有缓存项（包括过期项）
        
        Returns:
            缓存项字典
        """
        with self._lock:
            return self._cache.copy()
    
    def _evict_items(self, count: int) -> int:
        """
        移除最不常用的项
        
        Args:
            count: 要移除的项数量
            
        Returns:
            实际移除的项数量
        """
        if count <= 0 or not self._cache:
            return 0
        
        # 按访问次数和最后访问时间排序
        items = list(self._cache.values())
        items.sort(key=lambda x: (x.access_count, x.last_access_time))
        
        # 移除最不常用的项
        removed = 0
        for i in range(min(count, len(items))):
            if self.delete(items[i].key):
                removed += 1
        
        return removed
    
    def _cleanup_expired(self) -> int:
        """
        清理所有过期项
        
        Returns:
            清理的项数量
        """
        expired_keys = []
        for key, item in list(self._cache.items()):
            if item.is_expired():
                expired_keys.append(key)
        
        # 删除过期项
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)
    
    def _count_expired(self) -> int:
        """
        计算过期项数量
        
        Returns:
            过期项数量
        """
        count = 0
        for item in self._cache.values():
            if item.is_expired():
                count += 1
        return count
    
    def _auto_cleanup(self) -> None:
        """
        自动清理过期项
        """
        # 检查是否到达清理间隔
        now = time.time()
        if now - self.last_cleanup_time >= self.cleanup_interval:
            cleaned = self._cleanup_expired()
            self.last_cleanup_time = now
            if cleaned > 0:
                self.app_logger.debug(f"自动清理过期缓存项: {cleaned}项")
