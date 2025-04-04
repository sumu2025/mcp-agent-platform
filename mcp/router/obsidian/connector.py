"""
Obsidian连接器 - 基础的Obsidian知识库访问功能
"""

import os
import re
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

# 获取日志记录器
logger = logging.getLogger(__name__)


class ObsidianError(Exception):
    """Obsidian操作错误基类"""
    pass


class ObsidianConnector:
    """
    Obsidian连接器，提供基础的Obsidian知识库访问功能
    """
    
    def __init__(self, vault_path: Optional[str] = None):
        """
        初始化Obsidian连接器
        
        Args:
            vault_path: Obsidian仓库路径，如果不指定则尝试从环境变量或配置获取
        """
        # 获取仓库路径
        self.vault_path = vault_path or self._get_default_vault_path()
        
        # 验证仓库路径
        if not self._validate_vault_path():
            raise ObsidianError(f"无效的Obsidian仓库路径: {self.vault_path}")
        
        logger.info(f"初始化Obsidian连接器，仓库路径: {self.vault_path}")
    
    def _get_default_vault_path(self) -> str:
        """
        获取默认的Obsidian仓库路径
        
        Returns:
            默认仓库路径
        """
        # 尝试从环境变量获取
        vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
        
        if vault_path:
            return vault_path
        
        # 尝试从配置获取
        try:
            from ...utils.config import config
            vault_path = getattr(config, "OBSIDIAN_VAULT_PATH", None)
            if vault_path:
                return vault_path
        except ImportError:
            logger.warning("无法导入配置模块")
        
        # 返回一个默认值
        default_paths = [
            os.path.expanduser("~/Documents/Obsidian"),
            os.path.expanduser("~/Obsidian"),
            os.path.expanduser("~/Documents/obsidian-vault"),
            os.path.expanduser("~/obsidian-vault"),
        ]
        
        for path in default_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return path
        
        raise ObsidianError("无法找到默认的Obsidian仓库路径，请明确指定vault_path")
    
    def _validate_vault_path(self) -> bool:
        """
        验证仓库路径是否有效
        
        Returns:
            路径是否有效
        """
        vault_path = Path(self.vault_path)
        
        # 检查路径是否存在且是目录
        if not vault_path.exists() or not vault_path.is_dir():
            logger.error(f"仓库路径不存在或不是目录: {self.vault_path}")
            return False
        
        # 检查是否可以访问
        if not os.access(self.vault_path, os.R_OK | os.W_OK):
            logger.error(f"无法读写仓库路径: {self.vault_path}")
            return False
        
        # 检查是否包含.obsidian目录（这是一个Obsidian仓库的标志）
        obsidian_dir = vault_path / ".obsidian"
        if not obsidian_dir.exists() or not obsidian_dir.is_dir():
            logger.warning(f"仓库路径可能不是Obsidian仓库，未找到.obsidian目录: {self.vault_path}")
            # 不返回False，因为有些用户可能将仓库路径设为笔记目录而非Obsidian仓库根目录
        
        return True
    
    def list_notes(self, folder: Optional[str] = None, pattern: Optional[str] = None) -> List[str]:
        """
        列出仓库中的笔记
        
        Args:
            folder: 子文件夹路径（相对于仓库根目录）
            pattern: 文件名匹配模式（正则表达式）
            
        Returns:
            笔记路径列表（相对于仓库根目录）
        """
        base_path = Path(self.vault_path)
        if folder:
            base_path = base_path / folder
        
        if not base_path.exists() or not base_path.is_dir():
            logger.error(f"文件夹不存在: {base_path}")
            return []
        
        notes = []
        pattern_re = re.compile(pattern) if pattern else None
        
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".md"):
                    if pattern_re and not pattern_re.search(file):
                        continue
                    
                    rel_path = os.path.relpath(os.path.join(root, file), self.vault_path)
                    notes.append(rel_path)
        
        return notes
    
    def read_note(self, note_path: str) -> Optional[str]:
        """
        读取笔记内容
        
        Args:
            note_path: 笔记路径（相对于仓库根目录）
            
        Returns:
            笔记内容，如果笔记不存在则返回None
        """
        full_path = Path(self.vault_path) / note_path
        
        if not full_path.exists() or not full_path.is_file():
            logger.error(f"笔记不存在: {full_path}")
            return None
        
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"读取笔记失败: {full_path}, 错误: {str(e)}")
            return None
    
    def write_note(self, note_path: str, content: str, overwrite: bool = False) -> bool:
        """
        写入笔记内容
        
        Args:
            note_path: 笔记路径（相对于仓库根目录）
            content: 笔记内容
            overwrite: 是否覆盖已存在的笔记
            
        Returns:
            是否成功
        """
        full_path = Path(self.vault_path) / note_path
        
        # 检查目录是否存在，不存在则创建
        directory = full_path.parent
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"创建目录失败: {directory}, 错误: {str(e)}")
                return False
        
        # 检查文件是否已存在
        if full_path.exists() and not overwrite:
            logger.error(f"笔记已存在，未启用覆盖: {full_path}")
            return False
        
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"写入笔记成功: {full_path}")
            return True
        except Exception as e:
            logger.error(f"写入笔记失败: {full_path}, 错误: {str(e)}")
            return False
    
    def append_to_note(self, note_path: str, content: str, create_if_not_exists: bool = True) -> bool:
        """
        向笔记追加内容
        
        Args:
            note_path: 笔记路径（相对于仓库根目录）
            content: 要追加的内容
            create_if_not_exists: 如果笔记不存在，是否创建
            
        Returns:
            是否成功
        """
        full_path = Path(self.vault_path) / note_path
        
        # 检查目录是否存在，不存在则创建
        directory = full_path.parent
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"创建目录失败: {directory}, 错误: {str(e)}")
                return False
        
        # 检查文件是否存在
        if not full_path.exists():
            if create_if_not_exists:
                return self.write_note(note_path, content)
            else:
                logger.error(f"笔记不存在，未启用创建: {full_path}")
                return False
        
        try:
            with open(full_path, "a", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"追加笔记成功: {full_path}")
            return True
        except Exception as e:
            logger.error(f"追加笔记失败: {full_path}, 错误: {str(e)}")
            return False
    
    def get_daily_note_path(self, date: Optional[datetime.date] = None) -> str:
        """
        获取日记笔记路径
        
        Args:
            date: 日期，默认为今天
            
        Returns:
            日记笔记路径
        """
        date = date or datetime.date.today()
        date_str = date.strftime("%Y-%m-%d")
        
        return f"日记/{date_str}.md"
    
    def create_note_from_template(self, note_path: str, template_path: str, 
                                  variables: Optional[Dict[str, str]] = None) -> bool:
        """
        使用模板创建笔记
        
        Args:
            note_path: 目标笔记路径
            template_path: 模板笔记路径
            variables: 模板变量字典
            
        Returns:
            是否成功
        """
        # 读取模板
        template_content = self.read_note(template_path)
        if template_content is None:
            logger.error(f"模板不存在: {template_path}")
            return False
        
        # 替换变量
        if variables:
            for key, value in variables.items():
                placeholder = f"{{{{_{key}_}}}}"
                template_content = template_content.replace(placeholder, value)
        
        # 写入笔记
        return self.write_note(note_path, template_content)
    
    def search_notes(self, query: str, folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        简单搜索笔记内容
        
        Args:
            query: 搜索关键词
            folder: 搜索的子文件夹
            
        Returns:
            匹配的笔记列表，每个元素包含路径和匹配行
        """
        results = []
        notes = self.list_notes(folder)
        
        for note_path in notes:
            content = self.read_note(note_path)
            if content is None:
                continue
            
            lines = content.split("\n")
            matches = []
            
            for i, line in enumerate(lines):
                if query.lower() in line.lower():
                    matches.append({
                        "line_number": i + 1,
                        "line": line.strip()
                    })
            
            if matches:
                results.append({
                    "path": note_path,
                    "matches": matches
                })
        
        return results
