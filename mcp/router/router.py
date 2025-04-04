"""
智能路由器 - 整合任务分析和模型选择功能
"""

from typing import Dict, Optional, Any
import logging

from .analyzers import TaskAnalyzerManager, AnalysisResult
from .engines import WeightedModelSelector, SelectionResult

# 获取日志记录器
logger = logging.getLogger(__name__)


class Router:
    """
    智能路由器，负责分析用户输入并选择最适合的模型
    """
    
    def __init__(self):
        """
        初始化智能路由器
        """
        # 创建任务分析器管理器
        self.analyzer_manager = TaskAnalyzerManager()
        
        # 创建模型选择器
        self.model_selector = WeightedModelSelector()
        
        # 性能历史上下文（未来将从持久化存储加载）
        self.context = {
            "performance_history": {},
            "response_times": {}
        }
        
        logger.info("Router initialized")
    
    def route(self, text: str) -> Dict[str, Any]:
        """
        处理用户输入，返回路由结果
        
        Args:
            text: 用户输入文本
            
        Returns:
            路由结果字典，包含任务分析结果和模型选择结果
        """
        # 步骤1：分析任务类型
        analysis_result = self.analyzer_manager.analyze(text)
        logger.info(f"Task analysis result: {analysis_result.task_type.name} with confidence {analysis_result.confidence:.2f}")
        
        # 步骤2：选择合适的模型
        selection_result = self.model_selector.select(analysis_result, self.context)
        logger.info(f"Model selection result: {selection_result.provider}/{selection_result.model}")
        
        # 构建路由结果
        result = {
            "task_analysis": analysis_result.to_dict(),
            "model_selection": selection_result.to_dict(),
            "input_text": text[:100] + "..." if len(text) > 100 else text
        }
        
        return result
    
    def update_performance(self, task_type_name: str, provider: str, model: str, 
                         score: float, response_time: float) -> None:
        """
        更新模型性能历史
        
        Args:
            task_type_name: 任务类型名称
            provider: 提供商名称
            model: 模型名称
            score: 性能评分（0.0-1.0）
            response_time: 响应时间（秒）
        """
        # 更新性能历史
        if 'performance_history' not in self.context:
            self.context['performance_history'] = {}
        
        if task_type_name not in self.context['performance_history']:
            self.context['performance_history'][task_type_name] = {}
        
        if provider not in self.context['performance_history'][task_type_name]:
            self.context['performance_history'][task_type_name][provider] = {
                'samples': []
            }
        
        # 添加新样本
        self.context['performance_history'][task_type_name][provider]['samples'].append({
            'score': score,
            'model': model,
            'response_time': response_time
        })
        
        # 保留最近的样本（限制为10个）
        if len(self.context['performance_history'][task_type_name][provider]['samples']) > 10:
            self.context['performance_history'][task_type_name][provider]['samples'] = \
                self.context['performance_history'][task_type_name][provider]['samples'][-10:]
        
        # 更新响应时间
        if 'response_times' not in self.context:
            self.context['response_times'] = {}
        
        # 计算提供商的移动平均响应时间
        if provider not in self.context['response_times']:
            self.context['response_times'][provider] = response_time
        else:
            # 使用移动平均
            old_time = self.context['response_times'][provider]
            self.context['response_times'][provider] = old_time * 0.7 + response_time * 0.3
        
        logger.debug(f"Updated performance for {provider}/{model} on {task_type_name}: score={score:.2f}, time={response_time:.2f}s")
    
    def get_context(self) -> Dict[str, Any]:
        """
        获取当前上下文
        
        Returns:
            上下文字典
        """
        return self.context
