"""
内存向量存储 - 使用内存存储向量和索引，适合小型数据集
"""

import os
import json
import numpy as np
import pickle
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from pathlib import Path
import logging
import heapq

from .base import VectorStore, VectorStoreConfig, VectorRecord, SearchResult

# 设置日志
logger = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStore):
    """
    内存向量存储，使用内存存储向量和索引，适合小型数据集或原型设计
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化内存向量存储
        
        Args:
            config: 存储配置，如果为None则使用默认配置
        """
        super().__init__(config)
        
        # 存储向量记录的字典
        self.records: Dict[str, VectorRecord] = {}
        
        # 如果启用了标准化，则存储标准化后的向量，用于快速搜索
        self.normalized_vectors: Dict[str, np.ndarray] = {}
        
        # 支持的距离度量函数
        self.distance_functions = {
            "cosine": self._cosine_distance,
            "l2": self._l2_distance,
            "dot": self._dot_product,
        }
        
        # 验证配置
        if self.config.distance_metric not in self.distance_functions:
            raise ValueError(
                f"不支持的距离度量: {self.config.distance_metric}, "
                f"支持的度量: {', '.join(self.distance_functions.keys())}"
            )
        
        # 设置距离函数
        self.distance_function = self.distance_functions[self.config.distance_metric]
    
    def initialize(self) -> None:
        """初始化向量存储"""
        self.records = {}
        self.normalized_vectors = {}
        self._initialized = True
        logger.info(f"内存向量存储初始化完成，维度: {self.config.embedding_dim}, 距离度量: {self.config.distance_metric}")
    
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
        
        # 存储记录
        self.records[record.id] = record
        
        # 存储标准化向量
        self.normalized_vectors[record.id] = self._normalize_vector(record.embedding)
        
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
        
        record_ids = []
        for record in records:
            record_id = self.add(record)
            record_ids.append(record_id)
            
        return record_ids
    
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
        
        if len(self.records) == 0:
            return []
        
        # 验证查询向量维度
        if query_vector.shape[0] != self.config.embedding_dim:
            raise ValueError(
                f"查询向量维度不匹配: 查询维度 {query_vector.shape[0]}, "
                f"存储配置维度: {self.config.embedding_dim}"
            )
        
        # 标准化查询向量
        query_vector = self._normalize_vector(query_vector)
        
        # 计算所有向量的距离
        results = []
        for record_id, record in self.records.items():
            # 应用过滤条件
            if filter and not self._check_filter(record, filter):
                continue
                
            # 获取标准化向量
            normalized_vector = self.normalized_vectors[record_id]
            
            # 计算距离
            distance = self.distance_function(query_vector, normalized_vector)
            
            # 对于余弦距离和点积，我们需要转换为相似度（越高越好）
            if self.config.distance_metric in ["cosine", "dot"]:
                score = distance  # 对于这些度量，距离函数已经返回相似度
            else:
                # 对于L2等距离度量，将距离转换为相似度（越小越好）
                score = 1.0 / (1.0 + distance)
                
            # 创建搜索结果
            result = SearchResult(record=record, score=score)
            results.append(result)
        
        # 对结果排序（按相似度降序）
        results.sort(key=lambda x: x.score, reverse=True)
        
        # 返回前k个结果
        return results[:k]
    
    def delete(self, record_id: str) -> bool:
        """
        删除向量记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否成功删除
        """
        self.ensure_initialized()
        
        if record_id in self.records:
            # 删除记录和标准化向量
            del self.records[record_id]
            del self.normalized_vectors[record_id]
            return True
        
        return False
    
    def get(self, record_id: str) -> Optional[VectorRecord]:
        """
        获取向量记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            向量记录，如果不存在则为None
        """
        self.ensure_initialized()
        
        return self.records.get(record_id)
    
    def update(self, record: VectorRecord) -> bool:
        """
        更新向量记录
        
        Args:
            record: 更新后的向量记录
            
        Returns:
            是否成功更新
        """
        self.ensure_initialized()
        
        if record.id in self.records:
            # 验证向量维度
            if record.embedding.shape[0] != self.config.embedding_dim:
                raise ValueError(
                    f"向量维度不匹配: 记录维度 {record.embedding.shape[0]}, "
                    f"存储配置维度: {self.config.embedding_dim}"
                )
            
            # 更新记录和标准化向量
            self.records[record.id] = record
            self.normalized_vectors[record.id] = self._normalize_vector(record.embedding)
            return True
        
        return False
    
    def count(self) -> int:
        """
        获取存储的向量数量
        
        Returns:
            向量数量
        """
        self.ensure_initialized()
        
        return len(self.records)
    
    def clear(self) -> None:
        """
        清空向量存储
        """
        self.ensure_initialized()
        
        self.records.clear()
        self.normalized_vectors.clear()
        
        logger.info("内存向量存储已清空")
    
    def save(self) -> None:
        """
        保存向量存储
        
        如果配置了存储路径，则将数据保存到文件
        """
        self.ensure_initialized()
        
        if not self.config.storage_path:
            logger.warning("未配置存储路径，无法保存")
            return
        
        storage_path = Path(self.config.storage_path)
        os.makedirs(storage_path.parent, exist_ok=True)
        
        try:
            # 将记录转换为可序列化的形式
            serializable_records = {
                record_id: record.to_dict()
                for record_id, record in self.records.items()
            }
            
            # 保存数据
            with open(storage_path, 'wb') as f:
                pickle.dump(serializable_records, f)
                
            logger.info(f"内存向量存储已保存到 {storage_path}")
            
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
            raise
    
    def load(self) -> None:
        """
        加载向量存储
        
        如果配置了存储路径，则从文件加载数据
        """
        self.ensure_initialized()
        
        if not self.config.storage_path:
            logger.warning("未配置存储路径，无法加载")
            return
        
        storage_path = Path(self.config.storage_path)
        if not storage_path.exists():
            logger.warning(f"存储文件不存在: {storage_path}")
            return
        
        try:
            # 加载数据
            with open(storage_path, 'rb') as f:
                serializable_records = pickle.load(f)
            
            # 清空当前记录
            self.records.clear()
            self.normalized_vectors.clear()
            
            # 恢复记录
            for record_id, record_dict in serializable_records.items():
                record = VectorRecord.from_dict(record_dict)
                self.records[record_id] = record
                self.normalized_vectors[record_id] = self._normalize_vector(record.embedding)
                
            logger.info(f"内存向量存储已从 {storage_path} 加载，共 {len(self.records)} 条记录")
            
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            raise
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        获取存储信息
        
        Returns:
            存储信息字典
        """
        self.ensure_initialized()
        
        return {
            "type": "memory_vector_store",
            "count": len(self.records),
            "embedding_dim": self.config.embedding_dim,
            "distance_metric": self.config.distance_metric,
            "normalize_vectors": self.config.normalize_vectors,
            "storage_path": self.config.storage_path
        }
    
    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度 (0-1，越大越相似)
        """
        # 对于已标准化的向量，余弦相似度等于点积
        return np.dot(vec1, vec2)
    
    def _l2_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算L2距离
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            L2距离 (越小越相似)
        """
        return np.linalg.norm(vec1 - vec2)
    
    def _dot_product(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算点积
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            点积 (越大越相似)
        """
        return np.dot(vec1, vec2)


class FaissVectorStore(VectorStore):
    """
    Faiss向量存储，使用Facebook AI Similarity Search (Faiss) 库实现高效向量搜索
    
    注意: 使用此类需要安装Faiss库
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        初始化Faiss向量存储
        
        Args:
            config: 存储配置，如果为None则使用默认配置
        """
        # 设置默认索引类型为HNSW
        if not config:
            config = VectorStoreConfig(
                index_type="hnsw",
                distance_metric="l2",
                index_params={"M": 16, "efConstruction": 200}
            )
        
        super().__init__(config)
        
        # 存储向量记录的字典
        self.records: Dict[str, VectorRecord] = {}
        
        # 内部索引
        self.index = None
        
        # ID到索引的映射
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        # 当前索引大小
        self.current_index = 0
        
        # 标记删除的索引
        self.deleted_indices = set()
    
    def initialize(self) -> None:
        """初始化向量存储"""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "请安装Faiss库以使用FaissVectorStore: "
                "pip install faiss-cpu 或 pip install faiss-gpu"
            )
        
        import faiss
        
        # 创建Faiss索引
        dim = self.config.embedding_dim
        
        # 设置距离度量
        if self.config.distance_metric == "cosine":
            # 对于余弦相似度，使用内积，但需要规范化向量
            self.index = faiss.IndexFlatIP(dim)
            self.config.normalize_vectors = True
        elif self.config.distance_metric == "dot":
            # 对于点积，直接使用内积
            self.index = faiss.IndexFlatIP(dim)
        else:
            # 默认使用L2距离
            self.index = faiss.IndexFlatL2(dim)
        
        # 根据索引类型配置高级索引
        if self.config.index_type == "hnsw":
            # HNSW索引，适合大规模数据
            M = self.config.index_params.get("M", 16)
            efConstruction = self.config.index_params.get("efConstruction", 200)
            
            if self.config.distance_metric == "cosine" or self.config.distance_metric == "dot":
                hnsw_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            else:
                hnsw_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
                
            hnsw_index.hnsw.efConstruction = efConstruction
            hnsw_index.hnsw.efSearch = efConstruction
            self.index = hnsw_index
            
        elif self.config.index_type == "ivf":
            # IVF索引，适合大规模数据的快速搜索
            nlist = self.config.index_params.get("nlist", 100)
            
            if self.config.distance_metric == "cosine" or self.config.distance_metric == "dot":
                quantizer = faiss.IndexFlatIP(dim)
                ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(dim)
                ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
                
            # IVF索引需要训练，但在这里我们还没有数据，将在添加数据时处理
            self.index = ivf_index
            self.index_trained = False
        
        # 清空记录
        self.records = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.current_index = 0
        self.deleted_indices = set()
        
        self._initialized = True
        logger.info(
            f"Faiss向量存储初始化完成，维度: {dim}, "
            f"索引类型: {self.config.index_type}, "
            f"距离度量: {self.config.distance_metric}"
        )
    
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
        
        # 检查IVF索引是否需要训练
        if self.config.index_type == "ivf" and not getattr(self, "index_trained", False):
            if len(self.records) >= 100:  # 至少需要一些数据才能训练
                self._train_index()
        
        # 添加到Faiss索引
        vector_np = np.array([vector], dtype=np.float32)
        
        # 如果有删除的索引，重用它们
        if self.deleted_indices:
            index = self.deleted_indices.pop()
            self.index.remove_ids(np.array([index], dtype=np.int64))
        else:
            index = self.current_index
            self.current_index += 1
        
        self.index.add(vector_np)
        
        # 更新映射
        self.id_to_index[record.id] = index
        self.index_to_id[index] = record.id
        
        # 存储记录
        self.records[record.id] = record
        
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
        
        # 标准化向量
        vectors = np.array([
            self._normalize_vector(record.embedding)
            for record in records
        ], dtype=np.float32)
        
        # 检查IVF索引是否需要训练
        if self.config.index_type == "ivf" and not getattr(self, "index_trained", False):
            if len(self.records) + len(records) >= 100:  # 至少需要一些数据才能训练
                self._train_index(vectors)
        
        # 为每个记录分配索引
        indices = []
        for _ in records:
            if self.deleted_indices:
                index = self.deleted_indices.pop()
                # 需要从索引中移除，因为我们将在后面重新添加
                # 注意：这个操作在批量添加中可能效率较低
                self.index.remove_ids(np.array([index], dtype=np.int64))
            else:
                index = self.current_index
                self.current_index += 1
            indices.append(index)
        
        # 添加到Faiss索引
        self.index.add(vectors)
        
        # 更新映射和存储记录
        record_ids = []
        for record, index in zip(records, indices):
            record_id = record.id
            self.id_to_index[record_id] = index
            self.index_to_id[index] = record_id
            self.records[record_id] = record
            record_ids.append(record_id)
        
        return record_ids
    
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
        
        if len(self.records) == 0:
            return []
        
        # 验证查询向量维度
        if query_vector.shape[0] != self.config.embedding_dim:
            raise ValueError(
                f"查询向量维度不匹配: 查询维度 {query_vector.shape[0]}, "
                f"存储配置维度: {self.config.embedding_dim}"
            )
        
        # 标准化查询向量
        query_vector = self._normalize_vector(query_vector)
        query_np = np.array([query_vector], dtype=np.float32)
        
        # 如果有过滤条件，我们需要先获取符合条件的记录
        if filter:
            # 对于有过滤条件的情况，我们需要获取更多的结果，然后进行过滤
            # 这里我们获取至少k*2的结果，以增加找到足够符合条件的结果的可能性
            search_k = max(k * 2, 100)
            
            # 进行搜索
            distances, indices = self.index.search(query_np, search_k)
            
            # 过滤结果
            results = []
            for i, idx in enumerate(indices[0]):
                # 检查索引是否有效
                if idx == -1:
                    continue
                    
                # 获取记录ID
                record_id = self.index_to_id.get(int(idx))
                if not record_id:
                    continue
                    
                # 获取记录
                record = self.records.get(record_id)
                if not record:
                    continue
                    
                # 应用过滤条件
                if not self._check_filter(record, filter):
                    continue
                    
                # 计算相似度得分
                score = self._convert_distance_to_score(distances[0][i])
                
                # 创建搜索结果
                result = SearchResult(record=record, score=score, index=int(idx))
                results.append(result)
                
                # 如果已经有足够的结果，提前结束
                if len(results) >= k:
                    break
                    
            return results[:k]
            
        else:
            # 没有过滤条件，直接搜索
            distances, indices = self.index.search(query_np, k)
            
            # 处理结果
            results = []
            for i, idx in enumerate(indices[0]):
                # 检查索引是否有效
                if idx == -1:
                    continue
                    
                # 获取记录ID
                record_id = self.index_to_id.get(int(idx))
                if not record_id:
                    continue
                    
                # 获取记录
                record = self.records.get(record_id)
                if not record:
                    continue
                    
                # 计算相似度得分
                score = self._convert_distance_to_score(distances[0][i])
                
                # 创建搜索结果
                result = SearchResult(record=record, score=score, index=int(idx))
                results.append(result)
                
            return results
    
    def delete(self, record_id: str) -> bool:
        """
        删除向量记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            是否成功删除
        """
        self.ensure_initialized()
        
        if record_id in self.records:
            # 获取索引
            index = self.id_to_index[record_id]
            
            # 从Faiss索引中删除
            import faiss
            remove_indices = np.array([index], dtype=np.int64)
            
            try:
                self.index.remove_ids(remove_indices)
            except Exception as e:
                logger.warning(f"Faiss删除索引失败: {str(e)}")
            
            # 标记为删除
            self.deleted_indices.add(index)
            
            # 删除映射
            del self.id_to_index[record_id]
            del self.index_to_id[index]
            
            # 删除记录
            del self.records[record_id]
            
            return True
        
        return False
    
    def get(self, record_id: str) -> Optional[VectorRecord]:
        """
        获取向量记录
        
        Args:
            record_id: 记录ID
            
        Returns:
            向量记录，如果不存在则为None
        """
        self.ensure_initialized()
        
        return self.records.get(record_id)
    
    def update(self, record: VectorRecord) -> bool:
        """
        更新向量记录
        
        Args:
            record: 更新后的向量记录
            
        Returns:
            是否成功更新
        """
        self.ensure_initialized()
        
        if record.id in self.records:
            # 删除旧记录
            self.delete(record.id)
            
            # 添加新记录
            self.add(record)
            
            return True
        
        return False
    
    def count(self) -> int:
        """
        获取存储的向量数量
        
        Returns:
            向量数量
        """
        self.ensure_initialized()
        
        return len(self.records)
    
    def clear(self) -> None:
        """
        清空向量存储
        """
        self.ensure_initialized()
        
        # 重新初始化索引
        self.initialize()
        
        logger.info("Faiss向量存储已清空")
    
    def save(self) -> None:
        """
        保存向量存储
        
        如果配置了存储路径，则将数据保存到文件
        """
        self.ensure_initialized()
        
        if not self.config.storage_path:
            logger.warning("未配置存储路径，无法保存")
            return
        
        storage_path = Path(self.config.storage_path)
        os.makedirs(storage_path.parent, exist_ok=True)
        
        try:
            import faiss
            
            # 保存索引
            index_path = storage_path.with_suffix('.index')
            faiss.write_index(self.index, str(index_path))
            
            # 保存元数据
            metadata = {
                "records": {
                    record_id: record.to_dict()
                    for record_id, record in self.records.items()
                },
                "id_to_index": self.id_to_index,
                "index_to_id": {str(k): v for k, v in self.index_to_id.items()},  # 转换整数键为字符串
                "current_index": self.current_index,
                "deleted_indices": list(self.deleted_indices)
            }
            
            metadata_path = storage_path.with_suffix('.meta')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info(f"Faiss向量存储已保存到 {storage_path}")
            
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
            raise
    
    def load(self) -> None:
        """
        加载向量存储
        
        如果配置了存储路径，则从文件加载数据
        """
        self.ensure_initialized()
        
        if not self.config.storage_path:
            logger.warning("未配置存储路径，无法加载")
            return
        
        storage_path = Path(self.config.storage_path)
        index_path = storage_path.with_suffix('.index')
        metadata_path = storage_path.with_suffix('.meta')
        
        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"存储文件不存在: {index_path} 或 {metadata_path}")
            return
        
        try:
            import faiss
            
            # 加载索引
            self.index = faiss.read_index(str(index_path))
            
            # 加载元数据
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # 恢复记录
            self.records = {}
            for record_id, record_dict in metadata["records"].items():
                self.records[record_id] = VectorRecord.from_dict(record_dict)
                
            # 恢复映射
            self.id_to_index = metadata["id_to_index"]
            self.index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}  # 转换回整数键
            self.current_index = metadata["current_index"]
            self.deleted_indices = set(metadata["deleted_indices"])
            
            if self.config.index_type == "ivf":
                self.index_trained = True
                
            logger.info(f"Faiss向量存储已从 {storage_path} 加载，共 {len(self.records)} 条记录")
            
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            raise
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        获取存储信息
        
        Returns:
            存储信息字典
        """
        self.ensure_initialized()
        
        return {
            "type": "faiss_vector_store",
            "count": len(self.records),
            "embedding_dim": self.config.embedding_dim,
            "index_type": self.config.index_type,
            "distance_metric": self.config.distance_metric,
            "normalize_vectors": self.config.normalize_vectors,
            "storage_path": self.config.storage_path,
            "index_params": self.config.index_params
        }
    
    def _train_index(self, vectors: Optional[np.ndarray] = None) -> None:
        """
        训练IVF索引
        
        Args:
            vectors: 训练向量，如果为None则使用现有记录
        """
        import faiss
        
        if self.config.index_type != "ivf":
            return
            
        if vectors is None:
            # 收集所有向量
            vectors = np.array([
                self._normalize_vector(record.embedding)
                for record in self.records.values()
            ], dtype=np.float32)
        
        # 确保有足够的数据
        if len(vectors) < 100:
            logger.warning(f"训练数据不足，需要至少100个向量，当前只有 {len(vectors)} 个")
            return
        
        # 训练索引
        logger.info(f"训练IVF索引，使用 {len(vectors)} 个向量")
        self.index.train(vectors)
        self.index_trained = True
    
    def _convert_distance_to_score(self, distance: float) -> float:
        """
        将距离转换为相似度得分
        
        Args:
            distance: 距离值
            
        Returns:
            相似度得分 (0-1，越大越相似)
        """
        if self.config.distance_metric == "cosine" or self.config.distance_metric == "dot":
            # 对于余弦相似度和点积，距离越大越相似
            return distance
        else:
            # 对于L2距离，距离越小越相似，将其转换为相似度得分
            return 1.0 / (1.0 + distance)
