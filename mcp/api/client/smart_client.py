"""
智能API客户端 - 集成路由功能的高级客户端
"""

import logging
import time
from typing import Dict, Optional, Any, List, Union

from ...utils.config import config
from ...router import Router
from ...router.analyzers import TaskType
from ...router.obsidian import ObsidianRecorder
from ..base import GenerationParameters, GenerationResponse
from ..client import MCPClient

# 获取日志记录器
logger = logging.getLogger(__name__)


class SmartMCPClient(MCPClient):
    """
    智能MCP客户端，扩展基本客户端，增加智能路由和模型选择功能
    """
    
    def __init__(self, use_cache: bool = True, obsidian_recording: bool = False):
        """
        初始化智能MCP客户端
        
        Args:
            use_cache: 是否使用缓存
            obsidian_recording: 是否启用Obsidian记录
        """
        # 调用父类初始化
        super().__init__(use_cache=use_cache)
        
        # 创建路由器
        self.router = Router()
        
        # 是否启用智能路由
        self.smart_routing_enabled = True
        
        # 是否启用Obsidian记录
        self.obsidian_recording = obsidian_recording
        self.obsidian_recorder = None
        
        # 如果启用Obsidian记录，初始化记录器
        if self.obsidian_recording:
            try:
                self.obsidian_recorder = ObsidianRecorder()
                logger.info("Obsidian记录器初始化成功")
            except Exception as e:
                logger.error(f"Obsidian记录器初始化失败: {str(e)}")
                self.obsidian_recording = False
        
        logger.info("SmartMCPClient初始化")
    
    def generate(self, params: Union[GenerationParameters, Dict[str, Any]]) -> GenerationResponse:
        """
        智能生成文本，使用路由器选择最佳模型
        
        Args:
            params: 生成参数，可以是GenerationParameters或兼容的dict
            
        Returns:
            生成响应对象
        """
        # 将dict转换为GenerationParameters（如果需要）
        if isinstance(params, dict):
            params = GenerationParameters(**params)
        
        # 获取用户输入
        prompt = params.prompt
        
        # 路由信息
        route_result = None
        
        # 如果启用智能路由且未指定提供商，使用路由器选择模型
        if self.smart_routing_enabled and not params.provider:
            # 使用路由器分析并选择模型
            start_time = time.time()
            route_result = self.router.route(prompt)
            routing_time = time.time() - start_time
            
            logger.info(f"路由器选择提供商: {route_result['model_selection']['provider']} "
                       f"用时: {routing_time:.2f}秒")
            
            # 从路由结果中获取提供商和参数调整
            provider = route_result["model_selection"]["provider"]
            model = route_result["model_selection"]["model"]
            parameter_adjustments = route_result["model_selection"]["parameters"]
            
            # 调整生成参数
            # 注意：只调整未明确设置的参数
            if not params.model:
                params.model = model
            if not params.system:
                params.system = parameter_adjustments.get("system_prompt", "")
            
            # 仅当用户未明确设置时才应用推荐值
            if params.temperature is None or params.temperature == 0.7:  # 0.7是默认值
                params.temperature = parameter_adjustments.get("temperature", 0.7)
            if params.top_p is None or params.top_p == 0.9:  # 0.9是默认值
                params.top_p = parameter_adjustments.get("top_p", 0.9)
            
            # 记录参数调整
            logger.debug(f"调整参数: model={params.model}, "
                        f"temperature={params.temperature}, top_p={params.top_p}")
            
            # 设置提供商
            params.provider = provider
        
        # 记录API调用开始时间
        api_call_start = time.time()
        
        # 调用父类方法生成内容
        response = super().generate(params)
        
        # 计算API调用时间
        api_call_time = time.time() - api_call_start
        
        # 如果有路由结果，更新性能历史并保存路由信息
        if route_result:
            # 更新性能历史
            # 注意：这里我们使用一个简单的方法来模拟性能评分，实际应用中可能需要更复杂的评估方法
            task_type_name = route_result["task_analysis"]["task_type"]
            performance_score = 0.75  # 默认中等偏上
            
            # 记录性能数据
            self.router.update_performance(
                task_type_name=task_type_name,
                provider=params.provider,
                model=response.model,
                score=performance_score,
                response_time=api_call_time
            )
            
            # 在响应中添加路由信息
            response.routing_info = route_result
        
        # 如果启用Obsidian记录，记录生成内容
        if self.obsidian_recording and self.obsidian_recorder:
            try:
                # 记录生成内容
                self.obsidian_recorder.record_generation(
                    prompt=prompt,
                    response=response.text,
                    routing_info=getattr(response, "routing_info", None),
                    metadata={
                        "model": response.model,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens,
                        "response_time": api_call_time
                    }
                )
                
                # 如果有路由结果，记录性能数据
                if route_result:
                    self.obsidian_recorder.record_performance(
                        task_type=task_type_name,
                        provider=params.provider,
                        model=response.model,
                        score=performance_score,
                        response_time=api_call_time
                    )
                
                # 每10次生成后更新索引
                if hasattr(self, "_generation_count"):
                    self._generation_count += 1
                else:
                    self._generation_count = 1
                
                if self._generation_count % 10 == 0:
                    self.obsidian_recorder.create_index()
                
            except Exception as e:
                logger.error(f"Obsidian记录失败: {str(e)}")
        
        return response
    
    def set_routing_enabled(self, enabled: bool) -> None:
        """
        设置是否启用智能路由
        
        Args:
            enabled: 是否启用
        """
        self.smart_routing_enabled = enabled
        logger.info(f"智能路由{'启用' if enabled else '禁用'}")
    
    def set_obsidian_recording(self, enabled: bool) -> bool:
        """
        设置是否启用Obsidian记录
        
        Args:
            enabled: 是否启用
            
        Returns:
            是否成功
        """
        if enabled == self.obsidian_recording:
            return True
        
        if enabled:
            try:
                self.obsidian_recorder = ObsidianRecorder()
                self.obsidian_recording = True
                logger.info("Obsidian记录器初始化成功")
                return True
            except Exception as e:
                logger.error(f"Obsidian记录器初始化失败: {str(e)}")
                return False
        else:
            self.obsidian_recording = False
            logger.info("Obsidian记录已禁用")
            return True
    
    def get_routing_status(self) -> Dict[str, Any]:
        """
        获取智能路由状态信息
        
        Returns:
            路由状态信息
        """
        return {
            "enabled": self.smart_routing_enabled,
            "obsidian_recording": self.obsidian_recording,
            "context": self.router.get_context()
        }
    
    def get_task_types(self) -> List[Dict[str, Any]]:
        """
        获取所有支持的任务类型
        
        Returns:
            任务类型列表
        """
        task_types = []
        for task_type in TaskType:
            task_types.append({
                "name": task_type.name,
                "description": TaskType.get_description(task_type),
                "model_affinity": TaskType.get_model_affinity(task_type)
            })
        return task_types
