"""
嵌入缓存模块 - 管理嵌入的缓存，提高计算效率
"""

import os
import json
import numpy as np
import hashlib
import sqlite3
import time
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging

# 设置日志
logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    嵌入缓存抽象基类，定义缓存接口
    """
    
    def initialize(self) -> None:
        """初始化缓存"""
        pass
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的缓存嵌入
        
        Args:
            text: 文本内容
            
        Returns:
            嵌入向量，如果不存在则为None
        """
        raise NotImplementedError
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        设置文本的缓存嵌入
        
        Args:
            text: 文本内容
            embedding: 嵌入向量
        """
        raise NotImplementedError
    
    def contains(self, text: str) -> bool:
        """
        检查文本是否在缓存中
        
        Args:
            text: 文本内容
            
        Returns:
            是否存在于缓存中
        """
        raise NotImplementedError
    
    def clear(self) -> None:
        """清空缓存"""
        raise NotImplementedError
    
    def get_size(self) -> int:
        """
        获取缓存大小
        
        Returns:
            缓存中的嵌入数量
        """
        raise NotImplementedError
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            缓存信息字典
        """
        raise NotImplementedError
    
    def _hash_text(self, text: str) -> str:
        """
        计算文本的哈希值
        
        Args:
            text: 文本内容
            
        Returns:
            哈希值
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()


class MemoryEmbeddingCache(EmbeddingCache):
    """
    内存嵌入缓存，将嵌入存储在内存中
    """
    
    def __init__(self, max_size: int = 10000):
        """
        初始化内存缓存
        
        Args:
            max_size: 最大缓存条目数
        """
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def initialize(self) -> None:
        """初始化缓存"""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        logger.info(f"内存嵌入缓存初始化完成，最大大小: {self.max_size}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的缓存嵌入
        
        Args:
            text: 文本内容
            
        Returns:
            嵌入向量，如果不存在则为None
        """
        text_hash = self._hash_text(text)
        result = self.cache.get(text_hash)
        
        if result is not None:
            self.hits += 1
            return result
        else:
            self.misses += 1
            return None
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        设置文本的缓存嵌入
        
        Args:
            text: 文本内容
            embedding: 嵌入向量
        """
        text_hash = self._hash_text(text)
        
        # 如果缓存已满，移除最早的条目
        if len(self.cache) >= self.max_size:
            # 简单策略：移除第一个条目
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[text_hash] = embedding
    
    def contains(self, text: str) -> bool:
        """
        检查文本是否在缓存中
        
        Args:
            text: 文本内容
            
        Returns:
            是否存在于缓存中
        """
        text_hash = self._hash_text(text)
        return text_hash in self.cache
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("内存嵌入缓存已清空")
    
    def get_size(self) -> int:
        """
        获取缓存大小
        
        Returns:
            缓存中的嵌入数量
        """
        return len(self.cache)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            缓存信息字典
        """
        return {
            "type": "memory_cache",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


class DiskEmbeddingCache(EmbeddingCache):
    """
    磁盘嵌入缓存，将嵌入持久化存储在磁盘上
    """
    
    def __init__(self, 
                cache_dir: str,
                max_size: int = 100000,
                expiration: Optional[int] = None):
        """
        初始化磁盘缓存
        
        Args:
            cache_dir: 缓存目录
            max_size: 最大缓存条目数
            expiration: 缓存过期时间（秒），None表示永不过期
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.expiration = expiration
        self.hits = 0
        self.misses = 0
        self.index_file = self.cache_dir / "index.json"
        self.index = {}
    
    def initialize(self) -> None:
        """初始化缓存"""
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 加载索引
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
                logger.info(f"从 {self.index_file} 加载了 {len(self.index)} 个缓存索引")
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {str(e)}，将创建新索引")
                self.index = {}
        
        # 验证索引
        self._validate_index()
        
        # 保存索引
        self._save_index()
        
        logger.info(f"磁盘嵌入缓存初始化完成，目录: {self.cache_dir}, 条目: {len(self.index)}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的缓存嵌入
        
        Args:
            text: 文本内容
            
        Returns:
            嵌入向量，如果不存在则为None
        """
        text_hash = self._hash_text(text)
        
        if text_hash not in self.index:
            self.misses += 1
            return None
        
        # 检查是否过期
        if self.expiration is not None:
            entry_time = self.index[text_hash].get("time", 0)
            if time.time() - entry_time > self.expiration:
                # 缓存已过期
                self._remove_entry(text_hash)
                self.misses += 1
                return None
        
        # 加载嵌入
        try:
            embedding_file = self.cache_dir / f"{text_hash}.npy"
            if not embedding_file.exists():
                # 文件丢失
                self._remove_entry(text_hash)
                self.misses += 1
                return None
            
            embedding = np.load(embedding_file)
            self.hits += 1
            
            # 更新访问时间
            self.index[text_hash]["last_access"] = time.time()
            
            return embedding
            
        except Exception as e:
            logger.warning(f"加载嵌入失败: {str(e)}")
            self._remove_entry(text_hash)
            self.misses += 1
            return None
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        设置文本的缓存嵌入
        
        Args:
            text: 文本内容
            embedding: 嵌入向量
        """
        text_hash = self._hash_text(text)
        
        # 如果缓存已满，移除最早访问的条目
        if len(self.index) >= self.max_size:
            self._prune_cache()
        
        # 保存嵌入
        embedding_file = self.cache_dir / f"{text_hash}.npy"
        np.save(embedding_file, embedding)
        
        # 更新索引
        self.index[text_hash] = {
            "time": time.time(),
            "last_access": time.time(),
            "size": embedding.shape[0]
        }
        
        # 定期保存索引
        if len(self.index) % 100 == 0:
            self._save_index()
    
    def contains(self, text: str) -> bool:
        """
        检查文本是否在缓存中
        
        Args:
            text: 文本内容
            
        Returns:
            是否存在于缓存中
        """
        text_hash = self._hash_text(text)
        
        if text_hash not in self.index:
            return False
        
        # 检查是否过期
        if self.expiration is not None:
            entry_time = self.index[text_hash].get("time", 0)
            if time.time() - entry_time > self.expiration:
                return False
        
        # 检查文件是否存在
        embedding_file = self.cache_dir / f"{text_hash}.npy"
        return embedding_file.exists()
    
    def clear(self) -> None:
        """清空缓存"""
        # 删除所有嵌入文件
        for hash_val in self.index:
            embedding_file = self.cache_dir / f"{hash_val}.npy"
            if embedding_file.exists():
                try:
                    os.remove(embedding_file)
                except Exception as e:
                    logger.warning(f"删除嵌入文件失败: {str(e)}")
        
        # 清空索引
        self.index = {}
        self._save_index()
        
        logger.info("磁盘嵌入缓存已清空")
    
    def get_size(self) -> int:
        """
        获取缓存大小
        
        Returns:
            缓存中的嵌入数量
        """
        return len(self.index)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            缓存信息字典
        """
        # 计算缓存大小（字节）
        total_size = 0
        for file in os.listdir(self.cache_dir):
            if file.endswith(".npy"):
                file_path = os.path.join(self.cache_dir, file)
                total_size += os.path.getsize(file_path)
        
        return {
            "type": "disk_cache",
            "directory": str(self.cache_dir),
            "entries": len(self.index),
            "max_size": self.max_size,
            "expiration": self.expiration,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "disk_usage": total_size,
            "disk_usage_mb": total_size / (1024 * 1024)
        }
    
    def _save_index(self) -> None:
        """保存索引到磁盘"""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.warning(f"保存缓存索引失败: {str(e)}")
    
    def _validate_index(self) -> None:
        """验证索引，删除不存在的条目"""
        valid_index = {}
        for hash_val, info in self.index.items():
            embedding_file = self.cache_dir / f"{hash_val}.npy"
            if embedding_file.exists():
                valid_index[hash_val] = info
        
        removed = len(self.index) - len(valid_index)
        if removed > 0:
            logger.info(f"从索引中移除了 {removed} 个丢失的条目")
            
        self.index = valid_index
    
    def _remove_entry(self, text_hash: str) -> None:
        """
        移除一个缓存条目
        
        Args:
            text_hash: 文本哈希
        """
        if text_hash in self.index:
            del self.index[text_hash]
            
            embedding_file = self.cache_dir / f"{text_hash}.npy"
            if embedding_file.exists():
                try:
                    os.remove(embedding_file)
                except Exception as e:
                    logger.warning(f"删除嵌入文件失败: {str(e)}")
    
    def _prune_cache(self, amount: int = 100) -> None:
        """
        清理缓存，移除最早访问的条目
        
        Args:
            amount: 要移除的条目数量
        """
        if len(self.index) < amount:
            return
            
        # 按最后访问时间排序
        sorted_entries = sorted(
            self.index.items(),
            key=lambda x: x[1].get("last_access", 0)
        )
        
        # 移除最早访问的条目
        for i in range(min(amount, len(sorted_entries))):
            hash_val = sorted_entries[i][0]
            self._remove_entry(hash_val)
        
        logger.debug(f"清理缓存，移除了 {amount} 个最早访问的条目")


class SQLiteEmbeddingCache(EmbeddingCache):
    """
    SQLite嵌入缓存，将嵌入存储在SQLite数据库中
    """
    
    def __init__(self, 
                db_path: str,
                max_size: int = 100000,
                expiration: Optional[int] = None):
        """
        初始化SQLite缓存
        
        Args:
            db_path: 数据库文件路径
            max_size: 最大缓存条目数
            expiration: 缓存过期时间（秒），None表示永不过期
        """
        self.db_path = Path(db_path)
        self.max_size = max_size
        self.expiration = expiration
        self.hits = 0
        self.misses = 0
        self.conn = None
    
    def initialize(self) -> None:
        """初始化缓存"""
        # 创建数据库目录
        os.makedirs(self.db_path.parent, exist_ok=True)
        
        # 连接数据库
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # 创建表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            hash TEXT PRIMARY KEY,
            embedding BLOB,
            created_time REAL,
            last_access REAL,
            dim INTEGER
        )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_access ON embeddings (last_access)')
        
        # 提交变更
        self.conn.commit()
        
        # 清理过期缓存
        if self.expiration is not None:
            self._clean_expired()
        
        # 获取缓存大小
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        size = cursor.fetchone()[0]
        
        logger.info(f"SQLite嵌入缓存初始化完成，数据库: {self.db_path}, 条目: {size}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        获取文本的缓存嵌入
        
        Args:
            text: 文本内容
            
        Returns:
            嵌入向量，如果不存在则为None
        """
        if self.conn is None:
            self.initialize()
            
        text_hash = self._hash_text(text)
        cursor = self.conn.cursor()
        
        # 查询嵌入
        cursor.execute(
            'SELECT embedding, created_time FROM embeddings WHERE hash = ?',
            (text_hash,)
        )
        result = cursor.fetchone()
        
        if result is None:
            self.misses += 1
            return None
        
        # 检查是否过期
        if self.expiration is not None:
            created_time = result[1]
            if time.time() - created_time > self.expiration:
                # 删除过期条目
                cursor.execute('DELETE FROM embeddings WHERE hash = ?', (text_hash,))
                self.conn.commit()
                self.misses += 1
                return None
        
        # 加载嵌入
        try:
            embedding_blob = result[0]
            embedding = pickle.loads(embedding_blob)
            
            # 更新访问时间
            cursor.execute(
                'UPDATE embeddings SET last_access = ? WHERE hash = ?',
                (time.time(), text_hash)
            )
            self.conn.commit()
            
            self.hits += 1
            return embedding
            
        except Exception as e:
            logger.warning(f"加载嵌入失败: {str(e)}")
            cursor.execute('DELETE FROM embeddings WHERE hash = ?', (text_hash,))
            self.conn.commit()
            self.misses += 1
            return None
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        设置文本的缓存嵌入
        
        Args:
            text: 文本内容
            embedding: 嵌入向量
        """
        if self.conn is None:
            self.initialize()
            
        text_hash = self._hash_text(text)
        cursor = self.conn.cursor()
        
        # 检查缓存大小
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        size = cursor.fetchone()[0]
        
        # 如果缓存已满，移除最早访问的条目
        if size >= self.max_size:
            self._prune_cache()
        
        # 准备嵌入数据
        now = time.time()
        embedding_blob = pickle.dumps(embedding)
        
        # 插入或更新嵌入
        cursor.execute(
            '''
            INSERT OR REPLACE INTO embeddings 
            (hash, embedding, created_time, last_access, dim) 
            VALUES (?, ?, ?, ?, ?)
            ''',
            (text_hash, embedding_blob, now, now, embedding.shape[0])
        )
        
        self.conn.commit()
    
    def contains(self, text: str) -> bool:
        """
        检查文本是否在缓存中
        
        Args:
            text: 文本内容
            
        Returns:
            是否存在于缓存中
        """
        if self.conn is None:
            self.initialize()
            
        text_hash = self._hash_text(text)
        cursor = self.conn.cursor()
        
        # 查询嵌入
        cursor.execute(
            'SELECT created_time FROM embeddings WHERE hash = ?',
            (text_hash,)
        )
        result = cursor.fetchone()
        
        if result is None:
            return False
        
        # 检查是否过期
        if self.expiration is not None:
            created_time = result[0]
            if time.time() - created_time > self.expiration:
                return False
        
        return True
    
    def clear(self) -> None:
        """清空缓存"""
        if self.conn is None:
            self.initialize()
            
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM embeddings')
        self.conn.commit()
        
        logger.info("SQLite嵌入缓存已清空")
    
    def get_size(self) -> int:
        """
        获取缓存大小
        
        Returns:
            缓存中的嵌入数量
        """
        if self.conn is None:
            self.initialize()
            
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        return cursor.fetchone()[0]
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            缓存信息字典
        """
        if self.conn is None:
            self.initialize()
            
        cursor = self.conn.cursor()
        
        # 获取条目数
        cursor.execute('SELECT COUNT(*) FROM embeddings')
        size = cursor.fetchone()[0]
        
        # 获取数据库大小
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        # 获取平均维度
        cursor.execute('SELECT AVG(dim) FROM embeddings')
        avg_dim = cursor.fetchone()[0]
        
        return {
            "type": "sqlite_cache",
            "database": str(self.db_path),
            "entries": size,
            "max_size": self.max_size,
            "expiration": self.expiration,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "db_size": db_size,
            "db_size_mb": db_size / (1024 * 1024),
            "avg_dim": avg_dim
        }
    
    def _clean_expired(self) -> None:
        """清理过期缓存"""
        if self.conn is None or self.expiration is None:
            return
            
        cursor = self.conn.cursor()
        expire_time = time.time() - self.expiration
        
        cursor.execute(
            'DELETE FROM embeddings WHERE created_time < ?',
            (expire_time,)
        )
        
        deleted = cursor.rowcount
        self.conn.commit()
        
        if deleted > 0:
            logger.info(f"清理过期缓存，移除了 {deleted} 个条目")
    
    def _prune_cache(self, amount: int = 100) -> None:
        """
        清理缓存，移除最早访问的条目
        
        Args:
            amount: 要移除的条目数量
        """
        if self.conn is None:
            return
            
        cursor = self.conn.cursor()
        
        cursor.execute(
            'DELETE FROM embeddings WHERE hash IN (SELECT hash FROM embeddings ORDER BY last_access ASC LIMIT ?)',
            (amount,)
        )
        
        deleted = cursor.rowcount
        self.conn.commit()
        
        logger.debug(f"清理缓存，移除了 {deleted} 个最早访问的条目")
    
    def __del__(self):
        """析构函数，关闭数据库连接"""
        if self.conn is not None:
            self.conn.close()
