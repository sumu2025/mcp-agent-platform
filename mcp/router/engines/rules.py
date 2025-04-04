"""
模型选择规则 - 定义不同的模型选择规则
"""

from typing import Dict, List, Any
import logging

from ..analyzers import AnalysisResult, TaskType
from .base import SelectionRule

# 获取日志记录器
logger = logging.getLogger(__name__)


class TaskAffinityRule(SelectionRule):
    """
    任务亲和度规则 - 根据任务类型对模型的亲和度选择模型
    """
    
    def __init__(self, weight: float = 0.7):
        """
        初始化任务亲和度规则
        
        Args:
            weight: 规则权重（0.0-1.0）
        """
        self.weight = weight
    
    def get_name(self) -> str:
        """
        获取规则名称
        
        Returns:
            规则名称
        """
        return "task_affinity"
    
    def get_description(self) -> str:
        """
        获取规则描述
        
        Returns:
            规则描述
        """
        return "根据任务类型对不同模型的亲和度选择模型"
    
    def get_weight(self) -> float:
        """
        获取规则权重
        
        Returns:
            规则权重（0.0-1.0）
        """
        return self.weight
    
    def apply(self, analysis_result: AnalysisResult, context: Dict[str, Any]) -> Dict[str, float]:
        """
        应用选择规则，为每个模型计算分数
        
        Args:
            analysis_result: 任务分析结果
            context: 上下文信息，包含性能历史等
            
        Returns:
            模型得分字典，键为模型标识，值为分数（0.0-1.0）
        """
        # 获取任务类型对各模型的亲和度
        model_affinity = TaskType.get_model_affinity(analysis_result.task_type)
        
        # 如果分析结果置信度不高，降低分数
        confidence_factor = min(1.0, analysis_result.confidence * 1.2)
        
        # 调整分数
        scores = {}
        for model, affinity in model_affinity.items():
            scores[model] = affinity * confidence_factor
        
        logger.debug(f"TaskAffinityRule scores: {scores}")
        return scores


class PerformanceHistoryRule(SelectionRule):
    """
    性能历史规则 - 根据模型在类似任务上的历史表现选择模型
    """
    
    def __init__(self, weight: float = 0.6, min_samples: int = 5):
        """
        初始化性能历史规则
        
        Args:
            weight: 规则权重（0.0-1.0）
            min_samples: 应用规则所需的最小样本数
        """
        self.weight = weight
        self.min_samples = min_samples
    
    def get_name(self) -> str:
        """
        获取规则名称
        
        Returns:
            规则名称
        """
        return "performance_history"
    
    def get_description(self) -> str:
        """
        获取规则描述
        
        Returns:
            规则描述
        """
        return "根据模型在类似任务上的历史表现选择模型"
    
    def get_weight(self) -> float:
        """
        获取规则权重
        
        Returns:
            规则权重（0.0-1.0）
        """
        return self.weight
    
    def apply(self, analysis_result: AnalysisResult, context: Dict[str, Any]) -> Dict[str, float]:
        """
        应用选择规则，为每个模型计算分数
        
        Args:
            analysis_result: 任务分析结果
            context: 上下文信息，包含性能历史等
            
        Returns:
            模型得分字典，键为模型标识，值为分数（0.0-1.0）
        """
        # 如果上下文中没有性能历史数据，返回空分数
        if not context or 'performance_history' not in context:
            logger.debug("No performance history available")
            return {}
        
        performance_history = context['performance_history']
        task_type = analysis_result.task_type
        
        # 从性能历史中提取每个模型在当前任务类型上的表现
        scores = {}
        
        # 如果有性能历史数据，根据历史数据计算分数
        if task_type.name in performance_history:
            for model, history in performance_history[task_type.name].items():
                if len(history['samples']) >= self.min_samples:
                    # 计算平均性能分数
                    avg_score = sum(sample['score'] for sample in history['samples']) / len(history['samples'])
                    scores[model] = min(1.0, avg_score)
        
        logger.debug(f"PerformanceHistoryRule scores: {scores}")
        return scores


class CostEfficiencyRule(SelectionRule):
    """
    成本效率规则 - 根据模型的成本效率选择模型
    """
    
    def __init__(self, weight: float = 0.4, cost_sensitivity: float = 0.5):
        """
        初始化成本效率规则
        
        Args:
            weight: 规则权重（0.0-1.0）
            cost_sensitivity: 成本敏感度（0.0-1.0），值越高越倾向于选择低成本模型
        """
        self.weight = weight
        self.cost_sensitivity = cost_sensitivity
        self._init_model_costs()
    
    def _init_model_costs(self):
        """
        初始化模型成本信息
        """
        # 模型相对成本（1.0为基准）
        self.model_costs = {
            "claude": 1.0,     # 基准成本
            "deepseek": 0.7,   # 相对便宜30%
            "mock": 0.0,       # 模拟模式无成本
        }
    
    def get_name(self) -> str:
        """
        获取规则名称
        
        Returns:
            规则名称
        """
        return "cost_efficiency"
    
    def get_description(self) -> str:
        """
        获取规则描述
        
        Returns:
            规则描述
        """
        return "根据模型的成本效率选择模型"
    
    def get_weight(self) -> float:
        """
        获取规则权重
        
        Returns:
            规则权重（0.0-1.0）
        """
        return self.weight
    
    def apply(self, analysis_result: AnalysisResult, context: Dict[str, Any]) -> Dict[str, float]:
        """
        应用选择规则，为每个模型计算分数
        
        Args:
            analysis_result: 任务分析结果
            context: 上下文信息，包含性能历史等
            
        Returns:
            模型得分字典，键为模型标识，值为分数（0.0-1.0）
        """
        # 计算成本效率分数（成本越低分数越高）
        scores = {}
        max_cost = max(self.model_costs.values())
        
        for model, cost in self.model_costs.items():
            if max_cost == 0:
                scores[model] = 1.0
            else:
                # 成本越低，分数越高
                normalized_cost = cost / max_cost
                scores[model] = 1.0 - (normalized_cost * self.cost_sensitivity)
        
        logger.debug(f"CostEfficiencyRule scores: {scores}")
        return scores


class ResponseTimeRule(SelectionRule):
    """
    响应时间规则 - 根据模型的响应时间选择模型
    """
    
    def __init__(self, weight: float = 0.3, time_sensitivity: float = 0.5):
        """
        初始化响应时间规则
        
        Args:
            weight: 规则权重（0.0-1.0）
            time_sensitivity: 时间敏感度（0.0-1.0），值越高越倾向于选择响应快的模型
        """
        self.weight = weight
        self.time_sensitivity = time_sensitivity
        self._init_model_times()
    
    def _init_model_times(self):
        """
        初始化模型响应时间信息
        """
        # 模型相对响应时间（1.0为基准）
        self.model_times = {
            "claude": 1.0,     # 基准响应时间
            "deepseek": 0.8,   # 相对快20%
            "mock": 0.1,       # 模拟模式很快
        }
    
    def get_name(self) -> str:
        """
        获取规则名称
        
        Returns:
            规则名称
        """
        return "response_time"
    
    def get_description(self) -> str:
        """
        获取规则描述
        
        Returns:
            规则描述
        """
        return "根据模型的响应时间选择模型"
    
    def get_weight(self) -> float:
        """
        获取规则权重
        
        Returns:
            规则权重（0.0-1.0）
        """
        return self.weight
    
    def apply(self, analysis_result: AnalysisResult, context: Dict[str, Any]) -> Dict[str, float]:
        """
        应用选择规则，为每个模型计算分数
        
        Args:
            analysis_result: 任务分析结果
            context: 上下文信息，包含性能历史等
            
        Returns:
            模型得分字典，键为模型标识，值为分数（0.0-1.0）
        """
        # 如果上下文中有响应时间数据，使用真实数据
        if context and 'response_times' in context:
            # 使用真实的响应时间数据
            response_times = context['response_times']
            scores = {}
            
            if response_times:
                max_time = max(response_times.values())
                
                for model, time in response_times.items():
                    if max_time == 0:
                        scores[model] = 1.0
                    else:
                        # 时间越短，分数越高
                        normalized_time = time / max_time
                        scores[model] = 1.0 - (normalized_time * self.time_sensitivity)
            
                logger.debug(f"ResponseTimeRule scores (from real data): {scores}")
                return scores
        
        # 否则使用预设的模型时间
        scores = {}
        max_time = max(self.model_times.values())
        
        for model, time in self.model_times.items():
            if max_time == 0:
                scores[model] = 1.0
            else:
                # 时间越短，分数越高
                normalized_time = time / max_time
                scores[model] = 1.0 - (normalized_time * self.time_sensitivity)
        
        logger.debug(f"ResponseTimeRule scores (from defaults): {scores}")
        return scores
