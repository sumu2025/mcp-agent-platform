# MCP智能体中台项目环境设置指南

本指南将帮助您设置MCP智能体中台项目的开发环境。

## 1. 创建并激活虚拟环境

### macOS/Linux

```bash
# 进入项目目录
cd /Users/peacock/Desktop/AI架构师工作室/04-项目/01-进行中/MCP集成/mcp_agent_platform

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### Windows

```bash
# 进入项目目录
cd \path\to\mcp_agent_platform

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate
```

## 2. 安装项目依赖

一旦虚拟环境激活后（命令行前面会出现`(venv)`），安装项目依赖：

```bash
# 以开发模式安装项目及其依赖
pip install -e ".[dev]"
```

这将安装项目本身和所有在`pyproject.toml`中定义的依赖项。

## 3. 配置环境变量

从模板创建实际的环境变量文件：

```bash
cp .env.example .env
```

然后编辑`.env`文件，填入您的API密钥和其他配置选项：

```
# Claude API配置
CLAUDE_API_KEY=your_claude_api_key_here

# DeepSeek API配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Obsidian配置（可选）
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
```

## 4. 验证安装

安装完成后，可以运行以下命令验证安装是否成功：

```bash
# 查看版本信息
mcp version

# 测试CLI功能
mcp generate "简单测试一下"
```

如果一切正常，您应该能看到版本信息和一条提示信息，告诉您API功能尚未实现。

## 5. 编辑器设置（可选）

如果您使用VSCode，可以创建以下工作区设置：

`.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--max-line-length=88"
    ]
}
```

## 常见问题

1. **找不到mcp命令**
   
   确保您已正确激活虚拟环境，并以开发模式(-e)安装了项目。

2. **导入错误**
   
   如果遇到导入错误，请确认您的项目结构是否与预期一致，并且已正确安装项目。

3. **API密钥问题**
   
   请确保您已在`.env`文件中正确设置了API密钥，并且文件位于项目根目录。
