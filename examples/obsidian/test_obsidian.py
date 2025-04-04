"""
Obsidian集成测试脚本

本脚本测试Obsidian连接器和记录器功能。
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp.router.obsidian import ObsidianConnector, ObsidianRecorder


def test_obsidian_connector():
    """测试Obsidian连接器的基本功能"""
    
    print("====== Obsidian连接器测试 ======\n")
    
    # 使用默认路径初始化连接器
    try:
        connector = ObsidianConnector()
        print(f"成功连接到Obsidian仓库: {connector.vault_path}")
    except Exception as e:
        print(f"连接失败: {str(e)}")
        print("请通过--vault参数指定Obsidian仓库路径")
        return
    
    # 测试列出笔记
    try:
        notes = connector.list_notes()
        print(f"\n仓库中的笔记数量: {len(notes)}")
        if notes:
            print("示例笔记:")
            for note in notes[:5]:
                print(f"  - {note}")
            if len(notes) > 5:
                print(f"  ... 共{len(notes)}个笔记")
    except Exception as e:
        print(f"列出笔记失败: {str(e)}")
    
    # 测试写入笔记
    try:
        test_folder = "MCP测试"
        test_note = f"{test_folder}/测试笔记.md"
        content = f"""# MCP测试笔记

这是一个由MCP智能体中台生成的测试笔记。

生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 测试内容

这是一个简单的Markdown内容。

- 项目1
- 项目2
- 项目3
"""
        
        if connector.write_note(test_note, content):
            print(f"\n成功写入测试笔记: {test_note}")
        else:
            print(f"\n写入测试笔记失败")
    except Exception as e:
        print(f"写入笔记失败: {str(e)}")


def test_obsidian_recorder():
    """测试Obsidian记录器功能"""
    
    print("\n====== Obsidian记录器测试 ======\n")
    
    try:
        # 初始化记录器
        recorder = ObsidianRecorder()
        print(f"成功初始化记录器，基础文件夹: {recorder.base_folder}")
        
        # 测试记录生成内容
        prompt = "解释什么是大型语言模型"
        response = """大型语言模型(LLM)是一种基于神经网络的AI模型，它通过大量文本数据训练，学习了语言的规律和知识。

这些模型能够理解上下文，生成连贯的文本，回答问题，并执行多种自然语言处理任务。

典型的大型语言模型包括GPT系列、Claude、DeepSeek等。它们的应用范围广泛，从内容创作到代码生成，从客户服务到教育辅助。"""
        
        routing_info = {
            "task_analysis": {
                "task_type": "KNOWLEDGE_QA",
                "description": "知识问答",
                "confidence": 0.85
            },
            "model_selection": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "confidence": 0.78,
                "parameters": {
                    "temperature": 0.4,
                    "top_p": 0.8,
                    "system_prompt": "你是一个知识问答助手，擅长回答各种知识性问题。请提供准确、全面的信息。"
                }
            }
        }
        
        metadata = {
            "model": "deepseek-chat",
            "prompt_tokens": 8,
            "completion_tokens": 120,
            "total_tokens": 128,
            "response_time": 1.25
        }
        
        if recorder.record_generation(prompt, response, routing_info=routing_info, metadata=metadata):
            print("成功记录生成内容")
        else:
            print("记录生成内容失败")
        
        # 测试记录性能数据
        if recorder.record_performance("KNOWLEDGE_QA", "deepseek", "deepseek-chat", 0.85, 1.25):
            print("成功记录性能数据")
        else:
            print("记录性能数据失败")
        
        # 测试创建索引
        if recorder.create_index():
            print("成功创建索引页面")
        else:
            print("创建索引页面失败")
    
    except Exception as e:
        print(f"记录器测试失败: {str(e)}")


def main():
    """主函数"""
    
    # 处理命令行参数
    args = sys.argv[1:]
    vault_path = None
    
    for i, arg in enumerate(args):
        if arg == "--vault" and i + 1 < len(args):
            vault_path = args[i + 1]
    
    # 如果指定了仓库路径，设置环境变量
    if vault_path:
        os.environ["OBSIDIAN_VAULT_PATH"] = vault_path
        print(f"使用指定的Obsidian仓库路径: {vault_path}")
    
    # 运行测试
    test_obsidian_connector()
    test_obsidian_recorder()
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()
