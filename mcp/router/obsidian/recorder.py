"""
Obsidian记录器 - 负责将生成内容记录到Obsidian
"""

import os
import logging
import datetime
import json
from pathlib import Path
from typing import Dict, Optional, Any, Union, List

from .connector import ObsidianConnector
from ...router.analyzers import TaskType

# 获取日志记录器
logger = logging.getLogger(__name__)


class ObsidianRecorder:
    """
    Obsidian记录器，负责将生成内容和路由信息记录到Obsidian
    """
    
    def __init__(self, connector: Optional[ObsidianConnector] = None, 
                 base_folder: str = "MCP生成内容"):
        """
        初始化Obsidian记录器
        
        Args:
            connector: Obsidian连接器，如果不指定则创建一个新的
            base_folder: 基础文件夹路径（相对于仓库根目录）
        """
        # 获取连接器
        self.connector = connector or ObsidianConnector()
        
        # 基础文件夹
        self.base_folder = base_folder
        
        # 创建基础目录结构
        self._ensure_folder_structure()
        
        logger.info(f"初始化Obsidian记录器，基础文件夹: {self.base_folder}")
    
    def _ensure_folder_structure(self) -> None:
        """
        确保文件夹结构存在
        """
        folders = [
            self.base_folder,
            f"{self.base_folder}/按日期",
            f"{self.base_folder}/按任务类型",
            f"{self.base_folder}/性能数据"
        ]
        
        for folder in folders:
            path = Path(self.connector.vault_path) / folder
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"创建文件夹: {folder}")
                except Exception as e:
                    logger.error(f"创建文件夹失败: {folder}, 错误: {str(e)}")
    
    def record_generation(self, prompt: str, response: str, 
                         task_type: Optional[str] = None,
                         routing_info: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录生成内容
        
        Args:
            prompt: 提示词
            response: 生成的响应
            task_type: 任务类型名称
            routing_info: 路由信息
            metadata: 其他元数据
            
        Returns:
            是否成功
        """
        # 创建记录内容
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # 准备元数据
        metadata = metadata or {}
        metadata["timestamp"] = f"{date_str} {time_str}"
        
        if routing_info:
            metadata["task_type"] = routing_info.get("task_analysis", {}).get("task_type", "UNKNOWN")
            metadata["task_description"] = routing_info.get("task_analysis", {}).get("description", "未知任务类型")
            metadata["selected_model"] = f"{routing_info.get('model_selection', {}).get('provider', 'unknown')}/{routing_info.get('model_selection', {}).get('model', 'unknown')}"
        elif task_type:
            metadata["task_type"] = task_type
            try:
                task_enum = TaskType[task_type]
                metadata["task_description"] = TaskType.get_description(task_enum)
            except (KeyError, ValueError):
                metadata["task_description"] = "未知任务类型"
        
        # 创建Markdown内容
        content = self._create_markdown_content(prompt, response, metadata)
        
        # 获取文件路径
        file_path = self._get_file_path(metadata)
        
        # 写入文件
        return self.connector.write_note(file_path, content)
    
    def _create_markdown_content(self, prompt: str, response: str, 
                                metadata: Dict[str, Any]) -> str:
        """
        创建Markdown内容
        
        Args:
            prompt: 提示词
            response: 生成的响应
            metadata: 元数据
            
        Returns:
            Markdown内容
        """
        # YAML前置元数据
        yaml_metadata = "---\n"
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                yaml_metadata += f"{key}: |\n  {json.dumps(value, ensure_ascii=False, indent=2).replace(chr(10), chr(10) + '  ')}\n"
            else:
                yaml_metadata += f"{key}: {value}\n"
        yaml_metadata += "---\n\n"
        
        # 标题
        title = f"# 生成内容 ({metadata.get('timestamp', '未知时间')})\n\n"
        
        # 提示词部分
        prompt_section = "## 提示词\n\n"
        prompt_section += f"```\n{prompt}\n```\n\n"
        
        # 响应部分
        response_section = "## 响应\n\n"
        response_section += f"{response}\n\n"
        
        # 元数据部分
        metadata_section = "## 元数据\n\n"
        metadata_section += "| 属性 | 值 |\n|-------|------|\n"
        for key, value in metadata.items():
            if key != "timestamp" and not isinstance(value, (list, dict)):
                metadata_section += f"| {key} | {value} |\n"
        
        # 路由信息
        routing_section = ""
        if "task_type" in metadata:
            routing_section = "\n## 路由信息\n\n"
            routing_section += f"- 任务类型: {metadata.get('task_description', '未知')}\n"
            if "selected_model" in metadata:
                routing_section += f"- 选择模型: {metadata.get('selected_model', '未知')}\n"
        
        # 组合内容
        content = yaml_metadata + title + prompt_section + response_section + metadata_section + routing_section
        
        return content
    
    def _get_file_path(self, metadata: Dict[str, Any]) -> str:
        """
        获取文件路径
        
        Args:
            metadata: 元数据
            
        Returns:
            文件路径
        """
        timestamp = metadata.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        date_str = timestamp.split()[0]
        time_str = timestamp.split()[1].replace(":", "-")
        
        task_type = metadata.get("task_type", "UNKNOWN")
        
        # 按日期的路径
        date_path = f"{self.base_folder}/按日期/{date_str}/{time_str}.md"
        
        # 创建本次记录
        return date_path
    
    def record_performance(self, task_type: str, provider: str, model: str, 
                          score: float, response_time: float,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录性能数据
        
        Args:
            task_type: 任务类型
            provider: 提供商
            model: 模型
            score: 性能评分
            response_time: 响应时间
            metadata: 其他元数据
            
        Returns:
            是否成功
        """
        # 获取日期
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # 创建记录条目
        entry = {
            "timestamp": f"{date_str} {time_str}",
            "task_type": task_type,
            "provider": provider,
            "model": model,
            "score": score,
            "response_time": response_time
        }
        
        if metadata:
            entry.update(metadata)
        
        # 转换为Markdown表格行
        line = f"| {entry['timestamp']} | {entry['task_type']} | {entry['provider']}/{entry['model']} | {entry['score']:.2f} | {entry['response_time']:.2f}s |\n"
        
        # 获取性能数据文件路径
        file_path = f"{self.base_folder}/性能数据/性能记录-{date_str}.md"
        
        # 检查文件是否存在
        full_path = Path(self.connector.vault_path) / file_path
        if not full_path.exists():
            # 创建新文件，包含表头
            header = "# MCP智能体中台性能记录\n\n"
            header += f"日期: {date_str}\n\n"
            header += "| 时间 | 任务类型 | 模型 | 评分 | 响应时间 |\n"
            header += "|-------|----------|-------|------|----------|\n"
            header += line
            
            return self.connector.write_note(file_path, header)
        else:
            # 追加到已有文件
            return self.connector.append_to_note(file_path, line)
    
    def create_index(self) -> bool:
        """
        创建索引页面
        
        Returns:
            是否成功
        """
        # 收集信息
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        
        # 创建索引内容
        index_content = "# MCP智能体中台生成内容索引\n\n"
        index_content += f"最后更新: {date_str}\n\n"
        
        # 按日期索引
        index_content += "## 按日期索引\n\n"
        date_folders = sorted(self.list_date_folders(), reverse=True)
        for folder in date_folders[:10]:  # 显示最近10天
            folder_name = os.path.basename(folder)
            index_content += f"- [[{folder}|{folder_name}]]\n"
        
        if len(date_folders) > 10:
            index_content += f"- ... 共{len(date_folders)}天 ...\n"
        
        # 按任务类型索引
        index_content += "\n## 按任务类型索引\n\n"
        for task_type in TaskType:
            task_name = task_type.name
            task_desc = TaskType.get_description(task_type)
            task_folder = f"{self.base_folder}/按任务类型/{task_name}"
            index_content += f"- [[{task_folder}|{task_desc}]]\n"
        
        # 性能数据
        index_content += "\n## 性能数据\n\n"
        index_content += f"- [[{self.base_folder}/性能数据|浏览性能数据]]\n"
        
        # 写入索引文件
        return self.connector.write_note(f"{self.base_folder}/索引.md", index_content)
    
    def list_date_folders(self) -> List[str]:
        """
        列出按日期的文件夹
        
        Returns:
            文件夹路径列表
        """
        base_path = Path(self.connector.vault_path) / f"{self.base_folder}/按日期"
        if not base_path.exists() or not base_path.is_dir():
            return []
        
        folders = []
        for item in base_path.iterdir():
            if item.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", item.name):
                rel_path = os.path.relpath(item, self.connector.vault_path)
                folders.append(rel_path)
        
        return folders
