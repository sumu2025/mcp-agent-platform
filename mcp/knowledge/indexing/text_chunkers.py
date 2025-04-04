"""
文本分块器 - 实现各种文本分块策略
"""

import re
import uuid
from typing import Dict, List, Optional, Any, Tuple
import hashlib

from .base import TextChunk, TextChunker


class RecursiveTextChunker(TextChunker):
    """
    递归文本分块器 - 按层级结构对文本进行智能分块
    
    特点:
    1. 按语义边界分块（标题、段落、句子）
    2. 递归分块策略
    3. 保留块之间的重叠以维持上下文连贯性
    4. 智能控制块大小
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 100,
                 separators: Optional[List[str]] = None,
                 min_chunk_size: int = 50,
                 max_chunk_size: int = 1000):
        """
        初始化递归分块器
        
        Args:
            chunk_size: 目标块大小（单位：字符）
            chunk_overlap: 块之间的重叠大小（单位：字符）
            separators: 分隔符列表，按优先级排序，默认为["\n\n", "\n", ". ", " ", ""]
            min_chunk_size: 最小块大小（单位：字符）
            max_chunk_size: 最大块大小（单位：字符）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # 标题模式
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
    def split(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        将文本分割为多个块
        
        Args:
            text: 要分割的文本
            metadata: 分块时要携带的元数据
            
        Returns:
            文本块列表
        """
        metadata = metadata or {}
        
        # 1. 解析Markdown结构
        document_structure = self._parse_document_structure(text)
        
        # 2. 递归分块
        chunks = []
        self._split_sections(document_structure, metadata, chunks)
        
        # 3. 处理过小的块（合并相邻的小块）
        chunks = self._merge_small_chunks(chunks)
        
        # 4. 确保每个块都有唯一ID
        for i, chunk in enumerate(chunks):
            if not chunk.chunk_id:
                # 使用文本的哈希作为ID
                text_hash = hashlib.md5(chunk.text.encode()).hexdigest()[:8]
                chunk.chunk_id = f"chunk_{text_hash}_{i}"
                
            # 更新索引信息
            chunk.chunk_index = i
            
            # 添加额外元数据
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
        return chunks
    
    def get_chunker_name(self) -> str:
        """
        获取分块器名称
        
        Returns:
            分块器名称
        """
        return "recursive_chunker"
    
    def _parse_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        解析文档结构，将文档分解为章节树
        
        Args:
            text: 文档文本
            
        Returns:
            章节结构列表
        """
        lines = text.split('\n')
        sections = []
        current_section = {"level": 0, "title": None, "content": [], "children": []}
        section_stack = [current_section]
        
        for line in lines:
            # 检查是否是标题行
            heading_match = self.heading_pattern.match(line)
            
            if heading_match:
                # 这是一个标题行
                level = len(heading_match.group(1))  # 标题级别（#的数量）
                title = heading_match.group(2)  # 标题文本
                
                # 根据标题级别调整章节栈
                while level <= section_stack[-1]["level"] and len(section_stack) > 1:
                    section_stack.pop()
                
                # 创建新章节
                new_section = {
                    "level": level,
                    "title": title,
                    "content": [],
                    "children": []
                }
                
                # 将新章节添加到上级章节的子节点
                section_stack[-1]["children"].append(new_section)
                section_stack.append(new_section)
                current_section = new_section
            else:
                # 这是普通内容行，添加到当前章节的内容
                current_section["content"].append(line)
        
        # 返回顶级节点的子节点列表
        return section_stack[0]["children"]
    
    def _split_sections(self, 
                      sections: List[Dict[str, Any]], 
                      metadata: Dict[str, Any],
                      chunks: List[TextChunk],
                      parent_path: str = "") -> None:
        """
        递归地分割章节
        
        Args:
            sections: 章节列表
            metadata: 元数据
            chunks: 输出的文本块列表
            parent_path: 父章节路径
        """
        for section in sections:
            # 构建章节路径
            section_path = f"{parent_path}/{section['title']}" if parent_path else section['title']
            
            # 章节元数据
            section_metadata = metadata.copy()
            section_metadata.update({
                "heading": section["title"],
                "heading_level": section["level"],
                "section_path": section_path
            })
            
            # 构建包含标题的内容
            heading_prefix = "#" * section["level"]
            heading_line = f"{heading_prefix} {section['title']}\n\n"
            
            # 处理章节内容
            content_text = ''
            if section["content"]:
                content_text = '\n'.join(section["content"])
            
            # 合并标题和内容
            full_section_text = heading_line + content_text
            
            if full_section_text.strip():
                # 按段落分块
                self._split_text(full_section_text, section_metadata, chunks)
            
            # 递归处理子章节
            if section["children"]:
                self._split_sections(section["children"], metadata, chunks, section_path)
    
    def _split_text(self, 
                   text: str, 
                   metadata: Dict[str, Any],
                   chunks: List[TextChunk]) -> None:
        """
        将文本按分隔符拆分为适当大小的块
        
        Args:
            text: 要分割的文本
            metadata: 元数据
            chunks: 输出的文本块列表
        """
        # 如果文本小于最小块大小，直接添加为一个块
        if len(text) <= self.min_chunk_size:
            chunks.append(TextChunk(text=text, metadata=metadata))
            return
            
        # 如果文本小于最大块大小，直接添加为一个块
        if len(text) <= self.max_chunk_size:
            chunks.append(TextChunk(text=text, metadata=metadata))
            return
        
        # 递归地分割文本
        self._recursive_split(text, metadata, chunks)
    
    def _recursive_split(self, 
                        text: str, 
                        metadata: Dict[str, Any],
                        chunks: List[TextChunk]) -> None:
        """
        递归地将文本分割为块
        
        Args:
            text: 要分割的文本
            metadata: 元数据
            chunks: 输出的文本块列表
        """
        # 如果文本长度小于目标大小，直接添加为一个块
        if len(text) <= self.chunk_size:
            chunks.append(TextChunk(text=text, metadata=metadata))
            return
        
        # 尝试使用不同的分隔符
        for separator in self.separators:
            # 如果分隔符为空，则按字符分割
            if not separator:
                mid_point = self.chunk_size
                first_chunk = text[:mid_point]
                second_chunk = text[mid_point - self.chunk_overlap:]
                
                # 创建块
                self._recursive_split(first_chunk, metadata, chunks)
                self._recursive_split(second_chunk, metadata, chunks)
                return
            
            # 使用分隔符分割文本
            splits = text.split(separator)
            
            # 如果分割后只有一个部分，尝试下一个分隔符
            if len(splits) == 1:
                continue
            
            # 重新组合分割部分，形成块
            current_chunk = []
            current_length = 0
            
            for split in splits:
                split_text = f"{split}{separator}" if separator else split
                
                # 如果添加这个部分会超过目标大小，创建一个新块
                if current_length + len(split_text) > self.chunk_size and current_chunk:
                    chunk_text = ''.join(current_chunk)
                    chunks.append(TextChunk(text=chunk_text, metadata=metadata))
                    
                    # 保留重叠部分
                    overlap_size = min(self.chunk_overlap, len(chunk_text))
                    if overlap_size > 0:
                        current_chunk = [chunk_text[-overlap_size:]]
                        current_length = overlap_size
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(split_text)
                current_length += len(split_text)
            
            # 添加最后一个块
            if current_chunk:
                chunk_text = ''.join(current_chunk)
                chunks.append(TextChunk(text=chunk_text, metadata=metadata))
            
            # 分割成功，返回
            return
            
        # 如果所有分隔符都失败，使用字符级分割
        mid_point = self.chunk_size
        first_chunk = text[:mid_point]
        second_chunk = text[mid_point - self.chunk_overlap:]
        
        # 创建块
        self._recursive_split(first_chunk, metadata, chunks)
        self._recursive_split(second_chunk, metadata, chunks)
    
    def _merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        合并相邻的小块
        
        Args:
            chunks: 文本块列表
            
        Returns:
            合并后的文本块列表
        """
        if not chunks:
            return []
            
        result = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # 如果当前块加上下一个块仍然小于目标大小，合并它们
            if len(current_chunk.text) + len(next_chunk.text) <= self.chunk_size:
                combined_text = current_chunk.text + '\n\n' + next_chunk.text
                
                # 合并元数据
                combined_metadata = current_chunk.metadata.copy() if current_chunk.metadata else {}
                
                # 创建新块
                current_chunk = TextChunk(
                    text=combined_text,
                    metadata=combined_metadata,
                    source_document=current_chunk.source_document
                )
            else:
                # 添加当前块并继续
                result.append(current_chunk)
                current_chunk = next_chunk
        
        # 添加最后一个块
        result.append(current_chunk)
        
        return result


class SentenceTransformerChunker(TextChunker):
    """
    基于SentenceTransformers的分块器 - 利用语义边界进行分块
    
    此分块器使用语言模型来识别语义边界，实现更智能的分块。
    需要安装sentence-transformers库。
    """
    
    def __init__(self, 
                 chunk_size: int = 384,
                 chunk_overlap: int = 50,
                 model_name: str = "paraphrase-MiniLM-L6-v2"):
        """
        初始化分块器
        
        Args:
            chunk_size: 目标块大小（单位：tokens）
            chunk_overlap: 块之间的重叠大小（单位：tokens）
            model_name: SentenceTransformers模型名称
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def split(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        将文本分割为多个块
        
        Args:
            text: 要分割的文本
            metadata: 分块时要携带的元数据
            
        Returns:
            文本块列表
        """
        metadata = metadata or {}
        
        # 1. 分割为句子
        sentences = self._split_into_sentences(text)
        
        # 2. 按语义边界组织句子
        chunks = self._group_sentences_by_semantics(sentences, metadata)
        
        # 3. 确保每个块都有唯一ID
        for i, chunk in enumerate(chunks):
            if not chunk.chunk_id:
                chunk.chunk_id = f"st_chunk_{i}_{uuid.uuid4().hex[:8]}"
                
            # 更新索引信息
            chunk.chunk_index = i
            
            # 添加额外元数据
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            
        return chunks
    
    def get_chunker_name(self) -> str:
        """
        获取分块器名称
        
        Returns:
            分块器名称
        """
        return "sentencetransformer_chunker"
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割为句子
        
        Args:
            text: 要分割的文本
            
        Returns:
            句子列表
        """
        # 简单的句子分割，可以根据需要进行优化
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        
        # 过滤空句子
        sentences = [s for s in sentences if s.strip()]
        
        return sentences
    
    def _group_sentences_by_semantics(self, 
                                     sentences: List[str],
                                     metadata: Dict[str, Any]) -> List[TextChunk]:
        """
        根据语义相似性将句子分组为块
        
        Args:
            sentences: 句子列表
            metadata: 元数据
            
        Returns:
            文本块列表
        """
        if not sentences:
            return []
            
        # 嵌入句子
        sentence_embeddings = self.model.encode(sentences)
        
        # 语义分割点
        split_points = []
        current_tokens = 0
        
        for i in range(1, len(sentences)):
            # 估计token数量（粗略估计）
            current_tokens += len(sentences[i-1].split())
            
            # 计算相邻句子的语义相似度
            similarity = self._cosine_similarity(sentence_embeddings[i-1], sentence_embeddings[i])
            
            # 如果块大小达到目标或语义相似度低，添加分割点
            if current_tokens >= self.chunk_size or similarity < 0.5:  # 0.5是一个阈值，可以调整
                split_points.append(i)
                current_tokens = 0
        
        # 根据分割点创建块
        chunks = []
        start = 0
        
        for end in split_points:
            chunk_text = ' '.join(sentences[start:end])
            
            # 创建块
            chunk = TextChunk(
                text=chunk_text,
                metadata=metadata
            )
            chunks.append(chunk)
            
            # 更新起点，保留重叠
            overlap_sentences = min(self.chunk_overlap // 10, end - start)  # 粗略估计
            start = max(0, end - overlap_sentences)
        
        # 添加最后一个块
        if start < len(sentences):
            chunk_text = ' '.join(sentences[start:])
            chunk = TextChunk(
                text=chunk_text,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度（0-1）
        """
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product != 0 else 0
