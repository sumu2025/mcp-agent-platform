# MCP代码生成器使用指南

MCP代码生成器是一个专业的代码生成工具，基于DeepSeek API，可以快速生成高质量的TypeScript/JavaScript代码。它是MCP智能体中台项目的重要组成部分，用于加速开发过程。

## 安装与设置

1. 首先确保已安装依赖：

```bash
cd /Users/peacock/Desktop/AI架构师工作室/04-项目/01-进行中/MCP集成/mcp_agent_platform
npm install dotenv
```

2. 添加执行权限并链接命令：

```bash
chmod +x ./bin/mcp-code-gen.js
npm link
```

或者直接运行设置脚本：

```bash
sh setup_code_gen.sh
```

3. 确保`.env`文件中包含`DEEPSEEK_API_KEY`：

```
DEEPSEEK_API_KEY=your_api_key
```

## 使用方法

### 生成组件

```bash
# 基本用法
mcp-code-gen component "UserManager" --desc="用户管理组件" --type=class --out=./user-manager.ts

# 从JSON规格生成
mcp-code-gen component "ModelSwitcher" --desc="从JSON文件读取规格" --json=./examples/modelSwitcherSpec.json --out=./mcp/core/model-switcher.ts
```

### 生成适配器

```bash
# 从JSON规格生成适配器
mcp-code-gen adapter "ClaudeToDeepSeekAdapter" --source=Claude --target=DeepSeek --sourceInterface=./examples/adapterSpec.json --targetInterface=./examples/adapterSpec.json --out=./mcp/adapters/claude-to-deepseek.ts
```

### 生成测试

```bash
# 为现有代码生成测试
mcp-code-gen test "./mcp/core/model-switcher.ts" --framework=jest --out=./tests/model-switcher.test.ts
```

### 生成API客户端

```bash
# 生成API客户端
mcp-code-gen api "DeepSeekAPI" --desc="DeepSeek API客户端" --endpoints=./examples/endpoints.json --out=./mcp/api/deepseek.ts
```

## 示例工作流

以下是一个典型的工作流程，展示如何使用代码生成器高效开发：

1. 设计组件规格：

```json
{
  "name": "ModelSwitcher",
  "description": "模型切换器组件，负责在Claude和DeepSeek之间智能切换",
  "type": "class",
  "interfaces": ["IModelSwitcher"],
  "methods": [
    "checkAvailability(provider: string): Promise<boolean>",
    "switchModel(provider: string): Promise<boolean>",
    "getCurrentModel(): string", 
    "generateResponse(prompt: string): Promise<string>"
  ]
}
```

2. 生成组件代码：

```bash
mcp-code-gen component "ModelSwitcher" --json=./modelSwitcherSpec.json --out=./mcp/core/model-switcher.ts
```

3. 生成测试代码：

```bash
mcp-code-gen test "./mcp/core/model-switcher.ts" --out=./tests/model-switcher.test.ts
```

4. 集成到项目中并根据需要调整

## 提示与技巧

1. **温度参数控制创造性**：
   - 低温度（0.1-0.3）：生成更确定性、稳定的代码
   - 高温度（0.7-0.9）：生成更创造性的解决方案

2. **语言选择**：
   - 默认为TypeScript，但可以通过`--lang`选项指定其他语言

3. **调整生成模型**：
   - 默认使用`deepseek-coder`模型，但可以通过`--model`选项选择其他模型

4. **批量生成**：
   - 可以创建包含多个组件规格的文件，然后使用脚本循环生成

## 故障排除

1. **API密钥问题**：
   - 确保`.env`文件包含有效的`DEEPSEEK_API_KEY`
   - 可以使用`--apiKey`选项直接提供API密钥

2. **生成的代码有错误**：
   - 尝试降低温度参数（`--temp=0.1`）
   - 提供更详细的组件规格
   - 对复杂组件，考虑拆分为更小的组件

3. **命令找不到**：
   - 确保执行了`npm link`
   - 或使用`npm run code-gen`代替直接调用命令

## 高级用法

1. **自定义模板**：
   - 可以在`mcp/templates/prompts.js`中修改提示词模板

2. **保存常用规格**：
   - 将常用的组件规格保存为JSON文件，以便重复使用

3. **集成到开发流程**：
   - 考虑创建自定义脚本，自动化组件生成过程

## 限制与注意事项

1. 生成的代码可能需要手动调整以适应特定项目需求
2. 复杂组件可能需要多次迭代才能达到期望质量
3. 代码生成器依赖DeepSeek API，需要稳定的网络连接
