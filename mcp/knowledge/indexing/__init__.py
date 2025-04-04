"""
文档处理与索引模块 - 负责解析、处理和索引各种文档
"""

from .base import (
    DocumentProcessor, 
    TextChunk, 
    TextChunker, 
    MetadataExtractor,
    DocumentProcessorRegistry
)

from .markdown_processor import MarkdownProcessor
from .text_chunkers import RecursiveTextChunker, SentenceTransformerChunker
from .metadata_extractors import MarkdownMetadataExtractor, ObsidianMetadataExtractor

__all__ = [
    'DocumentProcessor',
    'TextChunk',
    'TextChunker',
    'MetadataExtractor',
    'DocumentProcessorRegistry',
    'MarkdownProcessor',
    'RecursiveTextChunker',
    'SentenceTransformerChunker',
    'MarkdownMetadataExtractor',
    'ObsidianMetadataExtractor'
]
