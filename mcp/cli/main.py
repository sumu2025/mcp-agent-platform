"""
命令行主模块 - 提供命令行工具的主入口
"""

import typer
from typing import Optional
from rich.console import Console

from .commands.generate import (generate_command, task_types_command, 
                              routing_info_command, obsidian_command)
from .. import __version__

# 创建应用
app = typer.Typer(
    name="mcp",
    help="MCP智能体中台命令行工具",
    add_completion=False
)

# 创建控制台
console = Console()


@app.callback()
def callback() -> None:
    """
    MCP智能体中台命令行工具
    
    用于访问MCP智能体中台的功能，包括文本生成、智能路由等。
    """
    pass


@app.command()
def version() -> None:
    """
    显示版本信息
    """
    console.print(f"MCP智能体中台 v{__version__}")


# 添加生成命令
app.command(name="generate")(generate_command)

# 添加任务类型命令
app.command(name="task-types")(task_types_command)

# 添加路由信息命令
app.command(name="routing")(routing_info_command)

# 添加Obsidian管理命令
app.command(name="obsidian")(obsidian_command)


# 主入口
if __name__ == "__main__":
    app()
