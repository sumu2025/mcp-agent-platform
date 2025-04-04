"""
文档处理器基类 - 定义文档处理的通用接口
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


class TextChunk:
    """文本块，表示文档的一个部分"""
    
    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_id: Optional[str] = None,
        source_document: Optional[str] = None,
        page_number: Optional[int] = None,
        chunk_index: Optional[int] = None
    ):
        """
        初始化文本块
        
        Args:
            text: 文本内容
            metadata: 元数据字典
            chunk_id: 块ID
            source_document: 源文档路径
            page_number: 页码（适用于PDF等）
            chunk_index: 块在文档中的索引
        """
        self.text = text
        self.metadata = metadata or {}
        self.chunk_id = chunk_id
        self.source_document = source_document
        self.page_number = page_number
        self.chunk_index = chunk_index
    
    def __repr__(self) -> str:
        """文本块的字符串表示"""
        return f"TextChunk(id={self.chunk_id}, text={self.text[:50]}{'...' if len(self.text) > 50 else ''})"
    
    def to_dict(self) -> Dict[str, Any]:
        """将文本块转换为字典"""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextChunk':
        """从字典创建文本块"""
        return cls(
            text=data["text"],
            metadata=data.get("metadata"),
            chunk_id=data.get("chunk_id"),
            source_document=data.get("source_document"),
            page_number=data.get("page_number"),
            chunk_index=data.get("chunk_index")
        )


class DocumentProcessor(ABC):
    """
    文档处理器抽象基类，定义文档处理的通用接口
    """
    
    @abstractmethod
    def process(self, file_path: Union[str, Path]) -> List[TextChunk]:
        """
        处理文档，返回文本块列表
        
        Args:
            file_path: 文档路径
            
        Returns:
            文本块列表
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: Union[str, Path]) -> bool:
        """
        检查是否支持处理该文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            是否支持
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        提取文档元数据
        
        Args:
            file_path: 文档路径
            
        Returns:
            元数据字典
        """
        pass
    
    @abstractmethod
    def get_processor_name(self) -> str:
        """
        获取处理器名称
        
        Returns:
            处理器名称
        """
        pass


class TextChunker(ABC):
    """
    文本分块器抽象基类，定义文本分块的通用接口
    """
    
    @abstractmethod
    def split(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        将文本分割为多个块
        
        Args:
            text: 要分割的文本
            metadata: 分块时要携带的元数据
            
        Returns:
            文本块列表
        """
        pass
    
    @abstractmethod
    def get_chunker_name(self) -> str:
        """
        获取分块器名称
        
        Returns:
            分块器名称
        """
        pass


class MetadataExtractor(ABC):
    """
    元数据提取器抽象基类，定义元数据提取的通用接口
    """
    
    @abstractmethod
    def extract(self, text: str, file_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        从文本提取元数据
        
        Args:
            text: 文本内容
            file_path: 文件路径（可选）
            
        Returns:
            元数据字典
        """
        pass
    
    @abstractmethod
    def get_extractor_name(self) -> str:
        """
        获取提取器名称
        
        Returns:
            提取器名称
        """
        pass


class DocumentProcessorRegistry:
    """
    文档处理器注册表，管理各种文档处理器
    """
    
    def __init__(self):
        """初始化注册表"""
        self.processors: List[DocumentProcessor] = []
    
    def register(self, processor: DocumentProcessor) -> None:
        """
        注册一个处理器
        
        Args:
            processor: 要注册的处理器
        """
        self.processors.append(processor)
    
    def get_processor_for_file(self, file_path: Union[str, Path]) -> Optional[DocumentProcessor]:
        """
        获取适用于指定文件的处理器
        
        Args:
            file_path: 文件路径
            
        Returns:
            适用的处理器，如果没有则返回None
        """
        for processor in self.processors:
            if processor.supports(file_path):
                return processor
        return None
    
    def get_all_processors(self) -> List[DocumentProcessor]:
        """
        获取所有注册的处理器
        
        Returns:
            处理器列表
        """
        return self.processors
