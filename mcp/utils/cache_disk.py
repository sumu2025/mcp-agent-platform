"""
磁盘缓存模块 - 提供基于文件系统的缓存实现
"""

import json
import os
import shutil
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Set, Tuple

from mcp.utils.cache_base import BaseCache, CacheItem
from mcp.utils.config import Config
from mcp.utils.exceptions import CacheError


class DiskCache(BaseCache):
    """基于文件系统的缓存实现"""
    
    def __init__(
        self, 
        name: str = "disk",
        cache_dir: Optional[str] = None,
        max_size: int = 10000,
        cleanup_interval: int = 3600
    ):
        """
        初始化磁盘缓存
        
        Args:
            name: 缓存名称
            cache_dir: 缓存目录路径，如果为None则使用配置中的路径
            max_size: 最大缓存项数量
            cleanup_interval: 自动清理间隔（秒）
        """
        super().__init__(name)
        
        # 获取缓存目录
        config = Config()
        if cache_dir is None:
            cache_dir = config.get("cache_dir", "./cache")
        
        # 创建缓存目录
        self.cache_dir = os.path.join(os.path.abspath(cache_dir), name)
        self._ensure_cache_dir()
        
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.last_cleanup_time = time.time()
        
        # 加载索引
        self._load_index()
        
        self.app_logger.debug(f"创建磁盘缓存: {name}, 路径: {self.cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期则返回None
        """
        with self._lock:
            # 检查索引中是否存在该键
            if key not in self._index:
                self.record_miss()
                return None
            
            # 获取索引项
            index_item = self._index[key]
            
            # 检查是否过期
            if self._is_expired(index_item):
                self.delete(key)
                self.record_miss()
                return None
            
            # 尝试从文件加载缓存项
            try:
                item = self._load_item(key)
                if item is None:
                    self.delete(key)
                    self.record_miss()
                    return None
                
                # 更新访问记录
                self._update_access(key)
                self.record_hit()
                
                # 检查是否需要清理
                self._auto_cleanup()
                
                return item.value
            except Exception as e:
                self.app_logger.error(f"加载缓存项失败: {key}, 错误: {str(e)}")
                self.delete(key)
                self.record_miss()
                return None
    
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
            if len(self._index) >= self.max_size and key not in self._index:
                # 清理过期项
                self._cleanup_expired()
                
                # 如果仍然超过限制，则移除最不常用的项
                if len(self._index) >= self.max_size:
                    self._evict_items(1)
            
            # 保存缓存项到文件
            try:
                self._save_item(item)
                
                # 更新索引
                self._index[key] = {
                    "key": key,
                    "created_time": item.created_time,
                    "last_access_time": item.last_access_time,
                    "access_count": item.access_count,
                    "expire_time": item.expire_time,
                }
                
                # 保存索引
                self._save_index()
                
                # 检查是否需要清理
                self._auto_cleanup()
                
                return True
            except Exception as e:
                self.app_logger.error(f"保存缓存项失败: {key}, 错误: {str(e)}")
                return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            存在并删除成功返回True，不存在返回False
        """
        with self._lock:
            if key not in self._index:
                return False
            
            # 删除文件
            try:
                file_path = self._get_item_path(key)
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                self.app_logger.error(f"删除缓存文件失败: {key}, 错误: {str(e)}")
            
            # 从索引中删除
            del self._index[key]
            
            # 保存索引
            self._save_index()
            
            return True
    
    def exists(self, key: str) -> bool:
        """
        检查缓存键是否存在且未过期
        
        Args:
            key: 缓存键
            
        Returns:
            如果存在且未过期返回True，否则返回False
        """
        with self._lock:
            # 检查索引中是否存在该键
            if key not in self._index:
                return False
            
            # 检查是否过期
            if self._is_expired(self._index[key]):
                self.delete(key)
                return False
            
            # 检查文件是否存在
            file_path = self._get_item_path(key)
            if not os.path.exists(file_path):
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
            try:
                # 删除所有缓存文件
                for key in list(self._index.keys()):
                    self.delete(key)
                
                # 清空索引
                self._index.clear()
                self._save_index()
                
                # 重置计数器
                self.hit_count = 0
                self.miss_count = 0
                self.last_cleanup_time = time.time()
                
                return True
            except Exception as e:
                self.app_logger.error(f"清空缓存失败: {str(e)}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含统计信息的字典
        """
        with self._lock:
            stats = {
                "name": self.name,
                "type": "disk",
                "dir": self.cache_dir,
                "size": len(self._index),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_ratio": self.get_hit_ratio(),
                "expired_count": self._count_expired(),
                "last_cleanup_time": self.last_cleanup_time,
            }
            
            # 计算总文件大小
            try:
                total_size = sum(os.path.getsize(self._get_item_path(key)) 
                               for key in self._index.keys()
                               if os.path.exists(self._get_item_path(key)))
                stats["total_file_size"] = total_size
            except Exception:
                stats["total_file_size"] = -1
            
            return stats
    
    def get_keys(self) -> List[str]:
        """
        获取所有未过期的缓存键
        
        Returns:
            缓存键列表
        """
        with self._lock:
            keys = []
            for key, index_item in list(self._index.items()):
                if not self._is_expired(index_item):
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
            return len(self._index)
    
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存项元数据
        
        Args:
            key: 缓存键
            
        Returns:
            元数据字典，如果不存在则返回None
        """
        with self._lock:
            if not self.exists(key):
                return None
            
            try:
                item = self._load_item(key)
                if item is None:
                    return None
                return item.metadata
            except Exception:
                return None
    
    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            index_dir = os.path.join(self.cache_dir, "index")
            os.makedirs(index_dir, exist_ok=True)
            data_dir = os.path.join(self.cache_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
        except Exception as e:
            self.app_logger.error(f"创建缓存目录失败: {str(e)}")
            raise CacheError(f"创建缓存目录失败: {str(e)}")
    
    def _get_index_path(self) -> str:
        """获取索引文件路径"""
        return os.path.join(self.cache_dir, "index", "index.json")
    
    def _get_item_path(self, key: str) -> str:
        """
        获取缓存项文件路径
        
        Args:
            key: 缓存键
            
        Returns:
            文件路径
        """
        return os.path.join(self.cache_dir, "data", f"{key}.json")
    
    def _load_index(self) -> None:
        """加载索引文件"""
        index_path = self._get_index_path()
        if not os.path.exists(index_path):
            self._index = {}
            return
        
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        except Exception as e:
            self.app_logger.error(f"加载索引失败: {str(e)}")
            self._index = {}
    
    def _save_index(self) -> None:
        """保存索引文件"""
        index_path = self._get_index_path()
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f)
        except Exception as e:
            self.app_logger.error(f"保存索引失败: {str(e)}")
    
    def _load_item(self, key: str) -> Optional[CacheItem]:
        """
        加载缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存项，如果不存在则返回None
        """
        file_path = self._get_item_path(key)
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return CacheItem.from_dict(data)
        except Exception as e:
            self.app_logger.error(f"加载缓存项失败: {key}, 错误: {str(e)}")
            return None
    
    def _save_item(self, item: CacheItem) -> None:
        """
        保存缓存项
        
        Args:
            item: 缓存项
        """
        file_path = self._get_item_path(item.key)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(item.to_dict(), f)
        except Exception as e:
            self.app_logger.error(f"保存缓存项失败: {item.key}, 错误: {str(e)}")
            raise CacheError(f"保存缓存项失败: {item.key}, 错误: {str(e)}")
    
    def _is_expired(self, index_item: Dict[str, Any]) -> bool:
        """
        检查索引项是否过期
        
        Args:
            index_item: 索引项
            
        Returns:
            如果已过期返回True，否则返回False
        """
        expire_time = index_item.get("expire_time")
        if expire_time is None:
            return False
        return time.time() > expire_time
    
    def _update_access(self, key: str) -> None:
        """
        更新访问记录
        
        Args:
            key: 缓存键
        """
        if key in self._index:
            self._index[key]["last_access_time"] = time.time()
            self._index[key]["access_count"] = self._index[key].get("access_count", 0) + 1
            # 不立即保存索引，以减少IO操作
    
    def _evict_items(self, count: int) -> int:
        """
        移除最不常用的项
        
        Args:
            count: 要移除的项数量
            
        Returns:
            实际移除的项数量
        """
        if count <= 0 or not self._index:
            return 0
        
        # 按访问次数和最后访问时间排序
        items = [(k, v) for k, v in self._index.items()]
        items.sort(key=lambda x: (x[1].get("access_count", 0), x[1].get("last_access_time", 0)))
        
        # 移除最不常用的项
        removed = 0
        for i in range(min(count, len(items))):
            if self.delete(items[i][0]):
                removed += 1
        
        return removed
    
    def _cleanup_expired(self) -> int:
        """
        清理所有过期项
        
        Returns:
            清理的项数量
        """
        expired_keys = []
        for key, index_item in list(self._index.items()):
            if self._is_expired(index_item):
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
        for index_item in self._index.values():
            if self._is_expired(index_item):
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
            
            # 延迟保存索引，在清理后一次性保存
            self._save_index()
            
            if cleaned > 0:
                self.app_logger.debug(f"自动清理过期缓存项: {cleaned}项")
