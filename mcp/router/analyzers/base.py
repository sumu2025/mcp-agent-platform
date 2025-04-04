"""
任务分析器基类 - 定义任务分析器的通用接口和功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional, Any
import re
import logging

from .task_types import TaskType

# 获取日志记录器
logger = logging.getLogger(__name__)


class AnalysisResult:
    """
    任务分析结果，包含任务类型和置信度信息
    """
    
    def __init__(self, task_type: TaskType, confidence: float, 
                 analysis_details: Optional[Dict[str, Any]] = None):
        """
        初始化分析结果
        
        Args:
            task_type: 识别出的任务类型
            confidence: 置信度（0.0-1.0）
            analysis_details: 分析过程详情（可选）
        """
        self.task_type = task_type
        self.confidence = confidence
        self.analysis_details = analysis_details or {}
    
    def __str__(self) -> str:
        """
        返回分析结果的字符串表示
        """
        return f"任务类型: {TaskType.get_description(self.task_type)}, 置信度: {self.confidence:.2f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将分析结果转换为字典形式
        """
        return {
            "task_type": self.task_type.name,
            "description": TaskType.get_description(self.task_type),
            "confidence": self.confidence,
            "details": self.analysis_details,
            "model_affinity": TaskType.get_model_affinity(self.task_type),
            "parameter_adjustments": TaskType.get_parameter_adjustments(self.task_type)
        }


class TaskAnalyzer(ABC):
    """
    任务分析器抽象基类，定义所有任务分析器必须实现的接口
    """
    
    @abstractmethod
    def analyze(self, text: str) -> AnalysisResult:
        """
        分析输入文本，识别任务类型
        
        Args:
            text: 用户输入文本
            
        Returns:
            任务分析结果
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        获取分析器名称
        
        Returns:
            分析器名称
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        获取分析器描述
        
        Returns:
            分析器描述
        """
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """
        获取分析器优先级（数值越小优先级越高）
        
        Returns:
            分析器优先级
        """
        pass
