"""
MCP智能体中台 - 智能路由层

此模块提供任务分析、模型选择和性能跟踪功能，
实现根据任务特性智能选择最适合的模型。
"""

from .analyzers import TaskAnalyzer, AnalysisResult, TaskType, TaskAnalyzerManager
from .engines import (ModelSelector, SelectionRule, SelectionResult, 
                     WeightedModelSelector)
from .router import Router
from .obsidian import ObsidianConnector, ObsidianRecorder

__all__ = [
    # 主路由器
    'Router',
    
    # 任务分析组件
    'TaskAnalyzer',
    'AnalysisResult',
    'TaskType',
    'TaskAnalyzerManager',
    
    # 模型选择组件
    'ModelSelector',
    'SelectionRule',
    'SelectionResult',
    'WeightedModelSelector',
    
    # Obsidian集成组件
    'ObsidianConnector',
    'ObsidianRecorder',
    
    # 下面的组件将在实现后取消注释
    # 'PerformanceTracker',
]
