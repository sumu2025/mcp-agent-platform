"""
模型选择器 - 基于多个规则选择最适合的模型
"""

from typing import Dict, List, Optional, Any, Tuple
import logging

from ..analyzers import AnalysisResult, TaskType
from .base import ModelSelector, SelectionRule, SelectionResult
from .rules import TaskAffinityRule, PerformanceHistoryRule, CostEfficiencyRule, ResponseTimeRule

# 获取日志记录器
logger = logging.getLogger(__name__)


class WeightedModelSelector(ModelSelector):
    """
    加权模型选择器，根据多个规则的加权结果选择最适合的模型
    """
    
    def __init__(self, default_provider: str = "deepseek", default_model: str = "deepseek-chat"):
        """
        初始化加权模型选择器
        
        Args:
            default_provider: 默认提供商
            default_model: 默认模型
        """
        self.rules: List[SelectionRule] = []
        self.default_provider = default_provider
        self.default_model = default_model
        
        # 注册默认规则
        self._register_default_rules()
        
        # 定义可用的提供商和模型
        self.available_providers = {
            "claude": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "deepseek": ["deepseek-chat", "deepseek-coder"],
            "mock": ["mock-default"]
        }
        
        # 定义默认参数
        self.default_parameters = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000,
            "system_prompt": ""
        }
    
    def _register_default_rules(self):
        """
        注册默认的选择规则
        """
        # 添加任务亲和度规则（最高权重）
        self.add_rule(TaskAffinityRule(weight=0.7))
        
        # 添加性能历史规则
        self.add_rule(PerformanceHistoryRule(weight=0.6))
        
        # 添加成本效率规则
        self.add_rule(CostEfficiencyRule(weight=0.4))
        
        # 添加响应时间规则
        self.add_rule(ResponseTimeRule(weight=0.3))
    
    def add_rule(self, rule: SelectionRule) -> None:
        """
        添加选择规则
        
        Args:
            rule: 要添加的选择规则
        """
        self.rules.append(rule)
        logger.info(f"Added selection rule: {rule.get_name()} with weight {rule.get_weight()}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        移除选择规则
        
        Args:
            rule_name: 要移除的规则名称
            
        Returns:
            是否成功移除
        """
        for i, rule in enumerate(self.rules):
            if rule.get_name() == rule_name:
                self.rules.pop(i)
                logger.info(f"Removed selection rule: {rule_name}")
                return True
        
        logger.warning(f"Rule not found: {rule_name}")
        return False
    
    def get_rules(self) -> List[SelectionRule]:
        """
        获取所有选择规则
        
        Returns:
            选择规则列表
        """
        return self.rules
    
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
        # 使用空字典作为默认上下文
        context = context or {}
        
        # 应用所有规则，计算每个模型的总分
        model_scores = {}
        rule_results = {}
        
        # 收集每个规则的分数
        for rule in self.rules:
            try:
                rule_scores = rule.apply(analysis_result, context)
                rule_results[rule.get_name()] = rule_scores
                
                # 按权重累计分数
                for model, score in rule_scores.items():
                    if model not in model_scores:
                        model_scores[model] = 0.0
                    model_scores[model] += score * rule.get_weight()
            except Exception as e:
                logger.error(f"Error applying rule {rule.get_name()}: {str(e)}")
        
        # 规范化总分
        total_weight = sum(rule.get_weight() for rule in self.rules)
        if total_weight > 0:
            for model in model_scores:
                model_scores[model] /= total_weight
        
        # 如果没有得分，使用默认模型
        if not model_scores:
            logger.warning("No model scores available, using default model")
            return SelectionResult(
                provider=self.default_provider,
                model=self.default_model,
                parameters=self._get_parameters(analysis_result),
                confidence=0.5,
                selection_details={"reason": "No model scores available, using defaults"}
            )
        
        # 选择得分最高的模型
        provider, model, score = self._select_best_model(model_scores)
        
        # 创建备选模型列表
        alternative_models = []
        for alt_provider, alt_model, alt_score in self._get_alternative_models(model_scores, provider, model):
            alternative_models.append({
                "provider": alt_provider,
                "model": alt_model,
                "score": alt_score
            })
        
        # 准备选择详情
        selection_details = {
            "rule_results": rule_results,
            "final_scores": model_scores,
            "task_type": analysis_result.task_type.name,
            "task_confidence": analysis_result.confidence
        }
        
        # 获取调整后的参数
        parameters = self._get_parameters(analysis_result)
        
        logger.info(f"Selected model: {provider}/{model} with score {score:.2f}")
        
        return SelectionResult(
            provider=provider,
            model=model,
            parameters=parameters,
            confidence=score,
            alternative_models=alternative_models,
            selection_details=selection_details
        )
    
    def _select_best_model(self, model_scores: Dict[str, float]) -> Tuple[str, str, float]:
        """
        从得分字典中选择最佳模型
        
        Args:
            model_scores: 模型得分字典
            
        Returns:
            (提供商, 模型, 得分)元组
        """
        best_provider = self.default_provider
        best_model = self.default_model
        best_score = 0.0
        
        # 找出得分最高的模型
        for provider_name, score in model_scores.items():
            if score > best_score:
                best_score = score
                best_provider = provider_name
                
                # 使用该提供商的默认模型
                if provider_name in self.available_providers and self.available_providers[provider_name]:
                    best_model = self.available_providers[provider_name][0]
                else:
                    best_model = provider_name  # 如果没有特定模型，使用提供商名作为模型名
        
        return best_provider, best_model, best_score
    
    def _get_alternative_models(self, model_scores: Dict[str, float], 
                              best_provider: str, best_model: str, 
                              max_alternatives: int = 2) -> List[Tuple[str, str, float]]:
        """
        获取得分次高的备选模型
        
        Args:
            model_scores: 模型得分字典
            best_provider: 最佳提供商
            best_model: 最佳模型
            max_alternatives: 最大备选数量
            
        Returns:
            备选模型列表，每个元素为(提供商, 模型, 得分)元组
        """
        # 创建(提供商, 模型, 得分)列表
        provider_models = []
        for provider, score in model_scores.items():
            if provider != best_provider:  # 排除最佳模型
                if provider in self.available_providers and self.available_providers[provider]:
                    model = self.available_providers[provider][0]
                else:
                    model = provider
                provider_models.append((provider, model, score))
        
        # 按得分降序排序
        provider_models.sort(key=lambda x: x[2], reverse=True)
        
        # 返回前N个备选
        return provider_models[:max_alternatives]
    
    def _get_parameters(self, analysis_result: AnalysisResult) -> Dict[str, Any]:
        """
        根据任务类型获取调整后的参数
        
        Args:
            analysis_result: 任务分析结果
            
        Returns:
            调整后的参数字典
        """
        # 获取任务类型推荐的参数调整
        task_params = TaskType.get_parameter_adjustments(analysis_result.task_type)
        
        # 合并默认参数和任务特定参数
        parameters = self.default_parameters.copy()
        parameters.update(task_params)
        
        return parameters
