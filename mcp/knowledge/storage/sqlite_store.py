"""
SQLite向量存储 - 使用SQLite数据库持久化存储向量
"""

import os
import json
import numpy as np
import pickle
import sqlite3
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from pathlib import Path
import logging
import struct
import io
import glob

from .base import VectorStore, VectorStoreConfig, VectorRecord, SearchResult

# 设置日志
logger = logging.getLogger(__name__)


class SQLiteVectorStore(VectorStore):
    """
    SQLite向量存储，使用SQLite数据库持久化存储向量
    
    优点：
    - 不依赖外部服务
    - 持久化存储
    - 支持元数据过滤
    - 事务支持
    
    缺点：
    - 性能较内存存储慢
    - 不支持高级向量索引 (如HNSW)
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化SQLite向量存储
        
        Args:
            config: 存储配置，如果为None则使用默认配置
        """
        # 确保配置了存储路径
        if not config or not config.storage_path:
            if not config:
                config = VectorStoreConfig()
                
            # 设置默认存储路径
            if not config.storage_path:
                config.storage_path = os.path.join(
                    os.path.expanduser("~"),
                    ".mcp",
                    "vector_store",
                    "vectors.db"
                )
        
        super().__init__(config)
        
        # SQLite连接
        self.conn = None
        
        # 元数据JSON字段的索引字段
        self.metadata_indexes = []
    
    def initialize(self) -> None:
        """初始化向量存储"""
        # 创建存储目录
        storage_path = Path(self.config.storage_path)
        os.makedirs(storage_path.parent, exist_ok=True)
        
        # 连接数据库
        self.conn = sqlite3.connect(storage_path)
        
        # 启用外键约束
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # 创建向量表
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS vectors (
            id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            text TEXT,
            metadata JSON,
            created_at REAL NOT NULL
        )
        ''')
        
        # 添加元数据JSON索引
        self._setup_metadata_indexes()
        
        # 创建临时表用于K最近邻搜索
        self.conn.execute('''
        CREATE TEMPORARY TABLE IF NOT EXISTS temp_vectors (
            id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        )
        ''')
        
        # 提交更改
        self.conn.commit()
        
        self._initialized = True
        logger.info(f"SQLite向量存储初始化完成，路径: {storage_path}, 维度: {self.config.embedding_dim}")
    
    def add(self, record: VectorRecord) -> str:
        """
        添加单个向量记录
        
        Args:
            record: 向量记录
            
        Returns:
            记录ID
        """
        self.ensure_initialized()
        
        # 确保记录有ID
        if not record.id:
            record.id = self.generate_record_id()
        
        # 验证向量维度
        if record.embedding.shape[0] != self.config.embedding_dim:
            raise ValueError(
                f"向量维度不匹配: 记录维度 {record.embedding.shape[0]}, "
                f"存储配置维度: {self.config.embedding_dim}"
            )
        
        # 标准化向量
        vector = self._normalize_vector(record.embedding)
        
        # 将向量转换为二进制
        vector_bin = self._vector_to_binary(vector)
        
        # 将元数据转换为JSON
        metadata_json = json.dumps(record.metadata) if record.metadata else None
        
        # 插入记录
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                '''
                INSERT OR REPLACE INTO vectors (id, embedding, text, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    record.id,
                    vector_bin,
                    record.text,
                    metadata_json,
                    record.created_at
                )
            )
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"添加向量记录失败: {str(e)}")
            raise
        
        return record.id
    
    def add_batch(self, records: List[VectorRecord]) -> List[str]:
        """
        批量添加向量记录
        
        Args:
            records: 向量记录列表
            
        Returns:
            记录ID列表
        """
        self.ensure_initialized()
        
        if not records:
            return []
        
        # 确保所有记录都有ID
        for record in records:
            if not record.id:
                record.id = self.generate_record_id()
        
        # 验证向量维度
        for record in records:
            if record.embedding.shape[0] != self.config.embedding_dim:
                raise ValueError(
                    f"向量维度不匹配: 记录维度 {record.embedding.shape[0]}, "
                    f"存储配置维度: {self.config.embedding_dim}"
                )
        
        # 准备批量插入数据
        batch_data = []
        for record in records:
            # 标准化向量
            vector = self._normalize_vector(record.embedding)
            
            # 将向量转换为二进制
            vector_bin = self._vector_to_binary(vector)
            
            # 将元数据转换为JSON
            metadata_json = json.dumps(record.metadata) if record.metadata else None
            
            batch_data.append((
                record.id,
                vector_bin,
                record.text,
                metadata_json,
                record.created_at
            ))
        
        # 批量插入记录
        try:
            cursor = self.conn.cursor()
            cursor.executemany(
                '''
                INSERT OR REPLACE INTO vectors (id, embedding, text, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                ''',
                batch_data
            )
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"批量添加向量记录失败: {str(e)}")
            raise
        
        return [record.id for record in records]
    
    def search(self, 
              query_vector: np.ndarray, 
              k: int = 10, 
              filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        搜索相似向量
        
        Args:
            query_vector: 查询向量
            k: 返回结果数量
            filter: 元数据过滤条件
            
        Returns:
            搜索结果列表
        """
        self.ensure_initialized()
        
        # 验证查询向量维度
        if query_vector.shape[0] != self.config.embedding_dim:
            raise ValueError(
                f"查询向量维度不匹配: 查询维度 {query_vector.shape[0]}, "
                f"存储配置维度: {self.config.embedding_dim}"
            )
        
        # 标准化查询向量
        query_vector = self._normalize_vector(query_vector)
        
        # 构建查询SQL
        filter_conditions, filter_params = self._build_filter_conditions(filter)
        
        # 根据距离度量选择相似度计算方式
        if self.config.distance_metric == "cosine" or self.config.distance_metric == "dot":
            # 使用点积计算余弦相似度（对于已标准化的向量）
            similarity_func = self._dot_product_similarity_sql
        else:
            # 使用L2距离
            similarity_func = self._l2_distance_sql
        
        # 查询相似向量
        try:
            cursor = self.conn.cursor()
            
            # 创建表达式
            expression = similarity_func(query_vector)
            
            # 构建完整SQL
            sql = f'''
            SELECT v.id, v.embedding, v.text, v.metadata, v.created_at, {expression} AS similarity
            FROM vectors v
            '''
            
            # 添加过滤条件
            if filter_conditions:
                sql += f" WHERE {filter_conditions}"
                
            # 排序和限制结果数量
            if self.config.distance_metric == "cosine" or self.config.distance_metric == "dot":
                # 对于余弦相似度和点积，越大越相似
                sql += " ORDER BY similarity DESC"
            else:
                # 对于L2距离，越小越相似
                sql += " ORDER BY similarity ASC"
                
            sql += f" LIMIT {k}"
            
            # 执行查询
            cursor.execute(sql, filter_params)
            
            # 处理结果
            results = []
            for row in cursor.fetchall():
                record_id, embedding_bin, text, metadata_json, created_at, similarity = row
                
                # 解析数据
                embedding = self._binary_to_vector(embedding_bin)
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # 创建记录
                record = VectorRecord(
                    id=record_id,
                    embedding=embedding,
                    text=text,
                    metadata=metadata,
                    created_at=created_at
                )
                
                # 计算最终相似度得分
                if self.config.distance_metric == "l2":
                    # 对于L2距离，将距离转换为相似度得分
                    score = 1.0 / (1.0 + similarity)
                else:
                    # 对于余弦相似度和点积，直接使用
                    score = similarity
                
                # 创建搜索结果
                result = SearchResult(record=record, score=score)
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索向量失败: {str(e)}")
            raise
    
    def delete(self, record_id: str) -> bool:
        """
        删除向量记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否成功删除
        """
        self.ensure_initialized()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM vectors WHERE id = ?", (record_id,))
            self.conn.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"删除向量记录失败: {str(e)}")
            raise
    
    def get(self, record_id: str) -> Optional[VectorRecord]:
        """
        获取向量记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            向量记录，如果不存在则为None
        """
        self.ensure_initialized()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, embedding, text, metadata, created_at FROM vectors WHERE id = ?", 
                (record_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            record_id, embedding_bin, text, metadata_json, created_at = row
            
            # 解析数据
            embedding = self._binary_to_vector(embedding_bin)
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            # 创建记录
            return VectorRecord(
                id=record_id,
                embedding=embedding,
                text=text,
                metadata=metadata,
                created_at=created_at
            )
            
        except Exception as e:
            logger.error(f"获取向量记录失败: {str(e)}")
            raise
    
    def update(self, record: VectorRecord) -> bool:
        """
        更新向量记录
        
        Args:
            record: 更新后的向量记录
            
        Returns:
            是否成功更新
        """
        self.ensure_initialized()
        
        # 检查记录是否存在
        exists = self.get(record.id) is not None
        if not exists:
            return False
            
        # 使用add方法更新记录
        self.add(record)
        
        return True
    
    def count(self) -> int:
        """
        获取存储的向量数量
        
        Returns:
            向量数量
        """
        self.ensure_initialized()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM vectors")
            return cursor.fetchone()[0]
            
        except Exception as e:
            logger.error(f"获取向量数量失败: {str(e)}")
            raise
    
    def clear(self) -> None:
        """
        清空向量存储
        """
        self.ensure_initialized()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM vectors")
            self.conn.commit()
            
            logger.info("SQLite向量存储已清空")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"清空向量存储失败: {str(e)}")
            raise
    
    def save(self) -> None:
        """
        保存向量存储
        
        SQLite向量存储已经是持久化的，不需要额外的保存操作
        """
        self.ensure_initialized()
        
        # 提交所有未提交的更改
        self.conn.commit()
        
        logger.info("SQLite向量存储已保存")
    
    def load(self) -> None:
        """
        加载向量存储
        
        SQLite向量存储已经是持久化的，不需要额外的加载操作
        """
        self.ensure_initialized()
        
        # 尝试获取向量数量，验证数据库连接正常
        count = self.count()
        
        logger.info(f"SQLite向量存储已加载，共 {count} 条记录")
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        获取存储信息
        
        Returns:
            存储信息字典
        """
        self.ensure_initialized()
        
        # 获取数据库大小
        storage_path = Path(self.config.storage_path)
        db_size = storage_path.stat().st_size if storage_path.exists() else 0
        
        # 获取其他统计信息
        try:
            cursor = self.conn.cursor()
            
            # 获取记录数
            cursor.execute("SELECT COUNT(*) FROM vectors")
            count = cursor.fetchone()[0]
            
            # 获取最早和最新的记录时间
            cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM vectors")
            min_time, max_time = cursor.fetchone()
            
            # 获取元数据字段统计
            metadata_fields = set()
            cursor.execute("SELECT metadata FROM vectors WHERE metadata IS NOT NULL LIMIT 1000")
            for row in cursor.fetchall():
                if row[0]:
                    metadata = json.loads(row[0])
                    metadata_fields.update(metadata.keys())
                    
            return {
                "type": "sqlite_vector_store",
                "count": count,
                "embedding_dim": self.config.embedding_dim,
                "distance_metric": self.config.distance_metric,
                "normalize_vectors": self.config.normalize_vectors,
                "storage_path": str(self.config.storage_path),
                "db_size": db_size,
                "db_size_mb": db_size / (1024 * 1024),
                "min_time": min_time,
                "max_time": max_time,
                "metadata_fields": list(metadata_fields),
                "metadata_indexes": self.metadata_indexes
            }
            
        except Exception as e:
            logger.error(f"获取存储信息失败: {str(e)}")
            return {
                "type": "sqlite_vector_store",
                "error": str(e),
                "storage_path": str(self.config.storage_path),
                "embedding_dim": self.config.embedding_dim,
                "distance_metric": self.config.distance_metric
            }
    
    def add_metadata_index(self, field_path: str) -> None:
        """
        为元数据字段添加索引
        
        Args:
            field_path: 元数据字段路径，例如 "category" 或 "author.name"
        """
        self.ensure_initialized()
        
        # 检查是否已经有此索引
        if field_path in self.metadata_indexes:
            logger.info(f"元数据字段 {field_path} 已有索引")
            return
            
        # 创建索引
        try:
            cursor = self.conn.cursor()
            
            # 为了索引JSON字段，我们使用JSON提取表达式
            if '.' in field_path:
                # 对于嵌套字段，使用JSON路径表达式
                parts = field_path.split('.')
                extract_expr = f"$.{'.'.join(parts)}"
            else:
                # 对于顶级字段，使用简单的提取表达式
                extract_expr = f"$.{field_path}"
                
            # 创建索引名称（替换非字母数字字符）
            index_name = f"idx_metadata_{field_path.replace('.', '_')}"
            index_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in index_name)
            
            # 创建索引
            sql = f'''
            CREATE INDEX IF NOT EXISTS {index_name}
            ON vectors (json_extract(metadata, '{extract_expr}'))
            '''
            
            cursor.execute(sql)
            self.conn.commit()
            
            # 添加到索引列表
            self.metadata_indexes.append(field_path)
            
            logger.info(f"为元数据字段 {field_path} 创建了索引")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"创建元数据索引失败: {str(e)}")
            raise
    
    def optimize(self) -> None:
        """
        优化数据库，提高查询性能
        """
        self.ensure_initialized()
        
        try:
            cursor = self.conn.cursor()
            
            # 执行VACUUM，整理数据库文件
            cursor.execute("VACUUM")
            
            # 分析表，更新统计信息
            cursor.execute("ANALYZE vectors")
            
            # 优化索引
            cursor.execute("PRAGMA optimize")
            
            self.conn.commit()
            
            logger.info("SQLite向量存储已优化")
            
        except Exception as e:
            logger.error(f"优化数据库失败: {str(e)}")
            raise
    
    def backup(self, backup_path: Optional[str] = None) -> str:
        """
        备份数据库
        
        Args:
            backup_path: 备份文件路径，如果为None则使用默认路径
            
        Returns:
            备份文件路径
        """
        self.ensure_initialized()
        
        if not backup_path:
            # 创建默认备份路径
            storage_path = Path(self.config.storage_path)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = str(storage_path.parent / f"vectors_backup_{timestamp}.db")
            
        try:
            # 创建备份目录
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # 使用SQLite的备份API
            backup_conn = sqlite3.connect(backup_path)
            self.conn.backup(backup_conn)
            backup_conn.close()
            
            logger.info(f"SQLite向量存储已备份到 {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"备份数据库失败: {str(e)}")
            raise
    
    def restore(self, backup_path: str) -> None:
        """
        从备份中恢复数据库
        
        Args:
            backup_path: 备份文件路径
        """
        # 检查备份文件是否存在
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"备份文件不存在: {backup_path}")
            
        # 关闭当前连接
        if self.conn:
            self.conn.close()
            self.conn = None
            
        try:
            # 创建临时连接
            backup_conn = sqlite3.connect(backup_path)
            
            # 重新连接到目标数据库
            self.conn = sqlite3.connect(self.config.storage_path)
            
            # 使用SQLite的备份API恢复数据
            backup_conn.backup(self.conn)
            backup_conn.close()
            
            # 重新设置元数据索引
            self._setup_metadata_indexes()
            
            self._initialized = True
            
            logger.info(f"SQLite向量存储已从 {backup_path} 恢复")
            
        except Exception as e:
            logger.error(f"恢复数据库失败: {str(e)}")
            raise
    
    def _setup_metadata_indexes(self) -> None:
        """设置元数据JSON索引"""
        # 清空索引列表
        self.metadata_indexes = []
        
        # 查询现有索引
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA index_list(vectors)")
        
        for row in cursor.fetchall():
            index_name = row[1]
            
            # 检查是否是元数据索引
            if index_name.startswith("idx_metadata_"):
                # 获取字段路径
                field_path = index_name[len("idx_metadata_"):].replace('_', '.')
                self.metadata_indexes.append(field_path)
    
    def _vector_to_binary(self, vector: np.ndarray) -> bytes:
        """
        将向量转换为二进制表示
        
        Args:
            vector: 向量
            
        Returns:
            二进制表示
        """
        # 使用低级操作更高效地转换
        # 对于float32（32位浮点数）
        return vector.astype(np.float32).tobytes()
    
    def _binary_to_vector(self, binary: bytes) -> np.ndarray:
        """
        将二进制表示转换为向量
        
        Args:
            binary: 二进制表示
            
        Returns:
            向量
        """
        # 使用低级操作更高效地转换
        # 从字节转换回float32数组
        vector = np.frombuffer(binary, dtype=np.float32)
        return vector
    
    def _build_filter_conditions(self, filter: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        """
        构建元数据过滤条件
        
        Args:
            filter: 过滤条件
            
        Returns:
            (SQL条件字符串, 参数列表)
        """
        if not filter:
            return "", []
            
        conditions = []
        params = []
        
        for key, value in filter.items():
            # 处理嵌套字段 (field.subfield)
            if '.' in key:
                parts = key.split('.')
                extract_expr = f"$.{'.'.join(parts)}"
            else:
                extract_expr = f"$.{key}"
                
            # 对于不同类型的值，使用不同的条件
            if value is None:
                conditions.append(f"json_extract(metadata, '{extract_expr}') IS NULL")
            elif isinstance(value, (int, float, bool, str)):
                conditions.append(f"json_extract(metadata, '{extract_expr}') = ?")
                params.append(value)
            elif isinstance(value, list):
                placeholders = ','.join(['?'] * len(value))
                conditions.append(f"json_extract(metadata, '{extract_expr}') IN ({placeholders})")
                params.extend(value)
            else:
                # 对于复杂类型，尝试进行JSON匹配
                conditions.append(f"json_extract(metadata, '{extract_expr}') = ?")
                params.append(json.dumps(value))
                
        return " AND ".join(conditions), params
    
    def _dot_product_similarity_sql(self, query_vector: np.ndarray) -> str:
        """
        生成计算点积相似度的SQL表达式
        
        Args:
            query_vector: 查询向量
            
        Returns:
            SQL表达式
        """
        # 在SQLite中，我们可以使用自定义函数计算余弦相似度
        # 但这里我们使用一个近似方法，通过在应用层执行部分计算
        
        # 将查询向量转换为二进制
        query_bin = self._vector_to_binary(query_vector)
        
        # 注册自定义函数
        self.conn.create_function("dot_product", 1, lambda x: self._dot_product_blob(query_bin, x))
        
        return "dot_product(embedding)"
    
    def _l2_distance_sql(self, query_vector: np.ndarray) -> str:
        """
        生成计算L2距离的SQL表达式
        
        Args:
            query_vector: 查询向量
            
        Returns:
            SQL表达式
        """
        # 将查询向量转换为二进制
        query_bin = self._vector_to_binary(query_vector)
        
        # 注册自定义函数
        self.conn.create_function("l2_distance", 1, lambda x: self._l2_distance_blob(query_bin, x))
        
        return "l2_distance(embedding)"
    
    def _dot_product_blob(self, query_bin: bytes, vector_bin: bytes) -> float:
        """
        计算二进制表示的点积
        
        Args:
            query_bin: 查询向量的二进制表示
            vector_bin: 存储向量的二进制表示
            
        Returns:
            点积
        """
        # 转换二进制为向量
        query_vector = np.frombuffer(query_bin, dtype=np.float32)
        vector = np.frombuffer(vector_bin, dtype=np.float32)
        
        # 计算点积
        return float(np.dot(query_vector, vector))
    
    def _l2_distance_blob(self, query_bin: bytes, vector_bin: bytes) -> float:
        """
        计算二进制表示的L2距离
        
        Args:
            query_bin: 查询向量的二进制表示
            vector_bin: 存储向量的二进制表示
            
        Returns:
            L2距离
        """
        # 转换二进制为向量
        query_vector = np.frombuffer(query_bin, dtype=np.float32)
        vector = np.frombuffer(vector_bin, dtype=np.float32)
        
        # 计算L2距离
        return float(np.linalg.norm(query_vector - vector))
    
    def __del__(self):
        """析构函数，关闭数据库连接"""
        if self.conn:
            self.conn.close()
