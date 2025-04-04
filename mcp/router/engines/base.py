"""
模型选择引擎基类 - 定义模型选择引擎的通用接口和功能
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging

from ..analyzers import AnalysisResult, TaskType

# 获取日志记录器
logger = logging.getLogger(__name__)


class SelectionResult:
    """
    模型选择结果，包含选择的模型和参数调整信息
    """
    
    def __init__(self, 
                 provider: str, 
                 model: str, 
                 parameters: Dict[str, Any],
                 confidence: float,
                 alternative_models: Optional[List[Dict[str, Any]]] = None,
                 selection_details: Optional[Dict[str, Any]] = None):
        """
        初始化选择结果
        
        Args:
            provider: 提供商名称（如'claude', 'deepseek', 'mock'）
            model: 模型名称
            parameters: 调整后的模型参数
            confidence: 选择置信度（0.0-1.0）
            alternative_models: 可选的备选模型列表（可选）
            selection_details: 选择过程详情（可选）
        """
        self.provider = provider
        self.model = model
        self.parameters = parameters
        self.confidence = confidence
        self.alternative_models = alternative_models or []
        self.selection_details = selection_details or {}
    
    def __str__(self) -> str:
        """
        返回选择结果的字符串表示
        """
        return f"模型选择: {self.provider}/{self.model}, 置信度: {self.confidence:.2f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将选择结果转换为字典形式
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "alternative_models": self.alternative_models,
            "details": self.selection_details
        }


class SelectionRule(ABC):
    """
    模型选择规则抽象基类，定义单个选择规则的接口
    """
    
    @abstractmethod
    def apply(self, analysis_result: AnalysisResult, context: Dict[str, Any]) -> Dict[str, float]:
        """
        应用选择规则，为每个模型计算分数
        
        Args:
            analysis_result: 任务分析结果
            context: 上下文信息，包含性能历史等
            
        Returns:
            模型得分字典，键为模型标识，值为分数（0.0-1.0）
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        获取规则名称
        
        Returns:
            规则名称
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        获取规则描述
        
        Returns:
            规则描述
        """
        pass
    
    @abstractmethod
    def get_weight(self) -> float:
        """
        获取规则权重
        
        Returns:
            规则权重（0.0-1.0）
        """
        pass


class ModelSelector(ABC):
    """
    模型选择器抽象基类，定义所有模型选择器必须实现的接口
    """
    
    @abstractmethod
    def select(self, analysis_result: AnalysisResult, 
               context: Optional[Dict[str, Any]] = None) -> SelectionResult:
        """
        根据任务分析结果选择最适合的模型
        
        Args:
            analysis_result: 任务分析结果
            context: 上下文信息，包含性能历史等（可选）
            
        Returns:
            模型选择结果
        """
        pass
    
    @abstractmethod
    def add_rule(self, rule: SelectionRule) -> None:
        """
        添加选择规则
        
        Args:
            rule: 要添加的选择规则
        """
        pass
    
    @abstractmethod
    def remove_rule(self, rule_name: str) -> bool:
        """
        移除选择规则
        
        Args:
            rule_name: 要移除的规则名称
            
        Returns:
            是否成功移除
        """
        pass
    
    @abstractmethod
    def get_rules(self) -> List[SelectionRule]:
        """
        获取所有选择规则
        
        Returns:
            选择规则列表
        """
        pass
