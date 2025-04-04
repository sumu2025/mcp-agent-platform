"""
生成命令模块 - 提供文本生成命令行功能
"""

import sys
import time
import json
from typing import Optional, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ...api.client.smart_client import SmartMCPClient
from ...router.analyzers import TaskType

# 创建控制台
console = Console()


def generate_command(
    prompt: str = typer.Argument(..., help="生成提示词"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="模型提供商 (claude, deepseek, mock)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名称"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="温度参数 (0.0-1.0)"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="最大生成token数"),
    system_prompt: Optional[str] = typer.Option(None, "--system-prompt", "-s", help="系统提示词"),
    smart_routing: bool = typer.Option(True, "--smart-routing/--no-smart-routing", help="是否启用智能路由"),
    show_routing: bool = typer.Option(False, "--show-routing", help="显示路由详情"),
    record_to_obsidian: bool = typer.Option(False, "--record", "-r", help="记录到Obsidian"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件路径"),
    no_cache: bool = typer.Option(False, "--no-cache", help="禁用缓存"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细信息")
) -> None:
    """
    生成文本内容
    """
    # 显示生成信息
    console.print(f"[bold green]生成文本...[/]")
    if verbose:
        console.print(f"提示词: [italic]{prompt[:50]}{'...' if len(prompt) > 50 else ''}[/]")
    
    # 创建智能客户端
    client = SmartMCPClient(use_cache=not no_cache, obsidian_recording=record_to_obsidian)
    
    # 设置是否启用智能路由
    client.set_routing_enabled(smart_routing)
    
    # 准备参数
    params = {
        "prompt": prompt,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system": system_prompt
    }
    
    # 移除None值
    params = {k: v for k, v in params.items() if v is not None}
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 生成内容
        response = client.generate(params)
        
        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 显示结果
        console.print(Panel(
            Markdown(response.text),
            title="生成内容",
            expand=False
        ))
        
        # 显示路由信息
        if show_routing and hasattr(response, "routing_info"):
            routing_info = response.routing_info
            task_type = routing_info["task_analysis"]["task_type"]
            task_description = routing_info["task_analysis"]["description"]
            task_confidence = routing_info["task_analysis"]["confidence"]
            
            model_provider = routing_info["model_selection"]["provider"]
            model_name = routing_info["model_selection"]["model"]
            model_confidence = routing_info["model_selection"]["confidence"]
            
            console.print(Panel(
                f"任务类型: [bold]{task_description}[/] (置信度: {task_confidence:.2f})\n"
                f"选择模型: [bold]{model_provider}/{model_name}[/] (置信度: {model_confidence:.2f})\n"
                f"参数调整: 温度={routing_info['model_selection']['parameters'].get('temperature', 'N/A')}\n",
                title="智能路由信息"
            ))
        
        # 显示Obsidian记录信息
        if record_to_obsidian:
            console.print(f"[green]内容已记录到Obsidian[/]")
        
        # 显示统计信息
        if verbose:
            console.print(
                f"[dim]生成时间: {elapsed_time:.2f}秒[/]\n"
                f"[dim]提示词tokens: {response.prompt_tokens}[/]\n"
                f"[dim]完成tokens: {response.completion_tokens}[/]\n"
                f"[dim]总tokens: {response.total_tokens}[/]\n"
                f"[dim]模型: {response.model}[/]"
            )
        
        # 保存到文件
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            console.print(f"[green]内容已保存至: {output_file}[/]")
    
    except Exception as e:
        console.print(f"[bold red]生成失败: {str(e)}[/]")
        sys.exit(1)


def task_types_command(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细信息")
) -> None:
    """
    显示支持的任务类型
    """
    # 创建智能客户端
    client = SmartMCPClient()
    
    # 获取任务类型
    task_types = client.get_task_types()
    
    # 显示任务类型
    console.print("[bold green]支持的任务类型:[/]")
    
    for task_type in task_types:
        name = task_type["name"]
        description = task_type["description"]
        
        if verbose:
            # 显示详细信息，包括模型亲和度
            affinities = task_type["model_affinity"]
            best_model = max(affinities.items(), key=lambda x: x[1])[0]
            
            console.print(f"[bold]{name}[/]: {description}")
            console.print(f"  最佳模型: [italic]{best_model}[/]")
            console.print(f"  模型亲和度:")
            for model, score in affinities.items():
                console.print(f"    - {model}: {score:.2f}")
            console.print("")
        else:
            # 简要信息
            console.print(f"[bold]{name}[/]: {description}")


def routing_info_command(
    json_output: bool = typer.Option(False, "--json", help="以JSON格式输出")
) -> None:
    """
    显示智能路由状态和信息
    """
    # 创建智能客户端
    client = SmartMCPClient()
    
    # 获取路由状态
    status = client.get_routing_status()
    
    if json_output:
        # 输出JSON格式
        print(json.dumps(status, indent=2))
    else:
        # 友好输出
        console.print("[bold green]智能路由状态:[/]")
        console.print(f"启用状态: {'[green]已启用[/]' if status['enabled'] else '[red]已禁用[/]'}")
        console.print(f"Obsidian记录: {'[green]已启用[/]' if status.get('obsidian_recording', False) else '[red]已禁用[/]'}")
        
        if "context" in status:
            context = status["context"]
            
            if "performance_history" in context and context["performance_history"]:
                console.print("\n[bold]性能历史:[/]")
                for task_type, providers in context["performance_history"].items():
                    console.print(f"任务类型: [bold]{task_type}[/]")
                    for provider, history in providers.items():
                        samples = history["samples"]
                        if samples:
                            avg_score = sum(s["score"] for s in samples) / len(samples)
                            avg_time = sum(s["response_time"] for s in samples) / len(samples)
                            console.print(f"  • {provider}: 样本数={len(samples)}, 平均分数={avg_score:.2f}, 平均时间={avg_time:.2f}s")
            
            if "response_times" in context and context["response_times"]:
                console.print("\n[bold]响应时间:[/]")
                for provider, time in context["response_times"].items():
                    console.print(f"  • {provider}: {time:.2f}s")


def obsidian_command(
    enable: Optional[bool] = typer.Option(None, "--enable/--disable", help="启用/禁用Obsidian记录"),
    vault_path: Optional[str] = typer.Option(None, "--vault", help="Obsidian仓库路径"),
    create_index: bool = typer.Option(False, "--create-index", help="创建索引页面")
) -> None:
    """
    管理Obsidian集成
    """
    try:
        from ...router.obsidian import ObsidianConnector, ObsidianRecorder
        
        if vault_path:
            # 设置环境变量
            import os
            os.environ["OBSIDIAN_VAULT_PATH"] = vault_path
            console.print(f"[green]已设置Obsidian仓库路径: {vault_path}[/]")
        
        # 测试连接
        connector = ObsidianConnector()
        vault_path = connector.vault_path
        
        console.print(f"[green]成功连接到Obsidian仓库: {vault_path}[/]")
        
        # 创建记录器
        recorder = ObsidianRecorder(connector)
        
        # 如果指定了--create-index，创建索引
        if create_index:
            if recorder.create_index():
                console.print("[green]成功创建索引页面[/]")
            else:
                console.print("[red]创建索引页面失败[/]")
        
        # 创建智能客户端
        client = SmartMCPClient()
        
        # 如果指定了--enable/--disable，设置记录状态
        if enable is not None:
            if client.set_obsidian_recording(enable):
                console.print(f"[green]Obsidian记录{'启用' if enable else '禁用'}成功[/]")
            else:
                console.print(f"[red]Obsidian记录{'启用' if enable else '禁用'}失败[/]")
    
    except Exception as e:
        console.print(f"[bold red]Obsidian操作失败: {str(e)}[/]")
        sys.exit(1)
