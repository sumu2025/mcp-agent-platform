# DeepSeek开发辅助指南

本文档提供如何使用DeepSeek API辅助MCP智能体中台Obsidian集成开发的指南。

## 设置

### 获取API密钥

1. 访问[DeepSeek官网](https://www.deepseek.com)注册账号
2. 导航到API部分获取API密钥
3. 配置mcp-dev-assist工具：
   ```bash
   mcp-dev-assist --setup
   # 根据提示输入API密钥
   ```

### 工具安装

确保mcp-dev-assist工具已正确安装：

```bash
# 添加执行权限
chmod +x /path/to/mcp_agent_platform/bin/mcp-dev-assist.js

# 创建命令软链接（可选）
ln -s /path/to/mcp_agent_platform/bin/mcp-dev-assist.js /usr/local/bin/mcp-dev-assist

# 测试安装
mcp-dev-assist --help
```

## 使用场景

### 适用于DeepSeek的任务

DeepSeek适合处理以下类型的开发任务：

1. **代码生成**：生成模板代码、实现特定功能的函数、转换代码
2. **文档编写**：生成注释、API文档、使用指南
3. **代码解释**：解释复杂代码段的功能
4. **调试建议**：针对错误提供可能的解决方案
5. **重构建议**：提供代码优化和重构的建议

### 不适用于DeepSeek的任务

以下任务应优先使用Claude处理：

1. **关键架构决策**：影响整体设计的决策
2. **复杂推理**：需要深度推理的问题
3. **多轮交互设计**：需要连续上下文的复杂交互
4. **用户体验设计**：涉及主观判断的设计决策

## 使用方法

### 命令行用法

```bash
# 基本用法
mcp-dev-assist "为用户管理系统设计数据库模式"

# 生成代码
mcp-dev-assist "实现一个函数，用于解析Obsidian笔记的YAML前置元数据"

# 代码解释
mcp-dev-assist "解释以下代码的功能: [粘贴代码]"

# 调试帮助
mcp-dev-assist "解决以下错误: [粘贴错误信息]"
```

### 最佳实践

1. **明确任务**：提供清晰、具体的指令
2. **提供上下文**：包括相关背景信息和约束
3. **指定格式**：说明需要的输出格式（代码、说明等）
4. **迭代改进**：根据初始输出提出改进建议

### 示例提示

#### 代码生成
```
为Obsidian笔记创建一个解析器，支持以下功能：
1. 提取YAML前置元数据
2. 识别内部链接 [[...]]
3. 解析Markdown标题层级

使用TypeScript实现，遵循模块化设计原则。
```

#### 代码解释
```
解释以下函数的作用和工作原理：

function buildContextPrompt() {
  // [粘贴代码]
}

特别关注它如何处理项目元数据和会话摘要。
```

#### 调试帮助
```
我在运行以下代码时遇到错误：

// [粘贴代码]

错误信息：
[粘贴错误信息]

可能的原因是什么？如何修复？
```

## 工作流集成

### Claude与DeepSeek协作

1. **Claude**：架构设计、决策制定、复杂推理
2. **DeepSeek**：实现细节、代码生成、文档编写

### 开发流程

1. 使用Claude进行高层设计和决策
2. 使用DeepSeek辅助实现细节
3. 使用Claude评审整体设计和实现
4. 使用DeepSeek进行优化和完善

## 提示与技巧

1. **提示词工程**：DeepSeek对详细、结构化的提示反应更好
2. **代码上下文**：提供足够的上下文有助于生成更好的代码
3. **迭代改进**：先获取初始版本，然后提供具体反馈进行改进
4. **比较方案**：要求DeepSeek提供多个备选方案并比较优缺点

## 常见问题

### Q: DeepSeek无法连接怎么办？
A: 检查API密钥和网络连接，确认配置文件正确。

### Q: 生成的代码质量不高怎么办？
A: 提供更详细的要求和约束，指定具体的设计模式或风格。

### Q: 如何在两个AI之间传递上下文？
A: 使用项目元数据和会话摘要系统保存和加载关键信息。

## 附录：常用命令

```bash
# 获取帮助
mcp-dev-assist --help

# 配置API密钥
mcp-dev-assist --setup

# 清理缓存
mcp-dev-assist --clear-cache

# 测试连接
mcp-dev-assist --test-connection
```
