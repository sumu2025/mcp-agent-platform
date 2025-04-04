"""
任务分析器管理器 - 集成和管理多个任务分析器
"""

from typing import Dict, List, Optional, Any
import logging

from .base import TaskAnalyzer, AnalysisResult
from .task_types import TaskType
from .keyword_analyzer import KeywordTaskAnalyzer
from .pattern_analyzer import PatternTaskAnalyzer

# 获取日志记录器
logger = logging.getLogger(__name__)


class TaskAnalyzerManager:
    """
    任务分析器管理器，整合多个分析器的结果，提供最终的任务类型分析
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        """
        初始化任务分析器管理器
        
        Args:
            confidence_threshold: 置信度阈值，低于此值的分析结果将被标记为低置信度
        """
        self.analyzers: List[TaskAnalyzer] = []
        self.confidence_threshold = confidence_threshold
        
        # 注册默认分析器
        self._register_default_analyzers()
    
    def _register_default_analyzers(self):
        """
        注册默认的任务分析器
        """
        # 添加模式分析器
        self.add_analyzer(PatternTaskAnalyzer())
        
        # 添加关键词分析器
        self.add_analyzer(KeywordTaskAnalyzer())
    
    def add_analyzer(self, analyzer: TaskAnalyzer):
        """
        添加任务分析器
        
        Args:
            analyzer: 要添加的任务分析器
        """
        self.analyzers.append(analyzer)
        # 按优先级排序（优先级数值小的排在前面）
        self.analyzers.sort(key=lambda x: x.get_priority())
        logger.info(f"Added task analyzer: {analyzer.get_name()} with priority {analyzer.get_priority()}")
    
    def remove_analyzer(self, analyzer_name: str) -> bool:
        """
        移除任务分析器
        
        Args:
            analyzer_name: 要移除的分析器名称
            
        Returns:
            是否成功移除
        """
        for i, analyzer in enumerate(self.analyzers):
            if analyzer.get_name() == analyzer_name:
                self.analyzers.pop(i)
                logger.info(f"Removed task analyzer: {analyzer_name}")
                return True
        
        logger.warning(f"Analyzer not found: {analyzer_name}")
        return False
    
    def analyze(self, text: str) -> AnalysisResult:
        """
        分析输入文本，整合多个分析器的结果
        
        Args:
            text: 用户输入文本
            
        Returns:
            最终的任务分析结果
        """
        if not text.strip():
            logger.warning("Empty text provided for analysis")
            return AnalysisResult(
                TaskType.GENERAL_CHAT,
                0.5,
                {"reason": "Empty text provided"}
            )
        
        # 收集所有分析器的结果
        results: List[AnalysisResult] = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(text)
                results.append(result)
                logger.info(f"Analyzer {analyzer.get_name()} result: {result.task_type.name} with confidence {result.confidence:.2f}")
            except Exception as e:
                logger.error(f"Error in analyzer {analyzer.get_name()}: {str(e)}")
        
        if not results:
            logger.warning("No analysis results available")
            return AnalysisResult(
                TaskType.GENERAL_CHAT,
                0.5,
                {"reason": "No analyzer produced valid results"}
            )
        
        # 整合结果
        return self._merge_results(results)
    
    def _merge_results(self, results: List[AnalysisResult]) -> AnalysisResult:
        """
        整合多个分析器的结果
        
        Args:
            results: 多个分析器的分析结果
            
        Returns:
            整合后的最终结果
        """
        # 按置信度降序排序
        sorted_results = sorted(results, key=lambda x: x.confidence, reverse=True)
        
        # 取置信度最高的结果
        best_result = sorted_results[0]
        
        # 如果最佳结果的置信度不足，标记为低置信度
        if best_result.confidence < self.confidence_threshold:
            logger.info(f"Best result has low confidence: {best_result.confidence:.2f}")
            
            # 仍然返回最佳结果，但在分析详情中添加低置信度标记
            best_result.analysis_details["low_confidence"] = True
            best_result.analysis_details["confidence_threshold"] = self.confidence_threshold
        
        # 添加其他分析器的结果摘要
        other_results = [
            {
                "task_type": r.task_type.name,
                "confidence": r.confidence
            }
            for r in sorted_results[1:] if r.confidence > self.confidence_threshold * 0.8
        ]
        
        if other_results:
            best_result.analysis_details["alternative_analyses"] = other_results
        
        return best_result
    
    def get_analyzers_info(self) -> List[Dict[str, Any]]:
        """
        获取所有注册分析器的信息
        
        Returns:
            分析器信息列表
        """
        return [
            {
                "name": analyzer.get_name(),
                "description": analyzer.get_description(),
                "priority": analyzer.get_priority()
            }
            for analyzer in self.analyzers
        ]
