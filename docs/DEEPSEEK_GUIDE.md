# DeepSeek开发辅助使用指南

本文档介绍如何使用DeepSeek API作为开发辅助工具，以提高MCP智能体中台项目的开发效率。

## 1. 配置与安装

### 获取API密钥

1. 访问DeepSeek官方网站并注册账号
2. 在API管理页面创建API密钥
3. 保存密钥到安全位置

### 配置工具

使用以下命令配置`mcp-dev-assist`工具：

```bash
# 第一次使用会提示输入API密钥
cd /path/to/mcp_agent_platform
node ./bin/mcp-dev-assist.js "测试连接"
```

## 2. 使用场景指南

### 合适的使用场景

DeepSeek适合以下开发任务：

- **代码生成**：创建实现特定功能的代码片段
- **文档编写**：生成注释、README、用户指南等
- **问题排查**：分析错误信息并提供解决方案
- **设计咨询**：讨论架构和接口设计方案

### 不适合的场景

保留以下任务给Claude：

- **核心架构决策**：影响整个系统的重要决策
- **复杂设计讨论**：需要深入理解项目背景的设计对话
- **用户体验设计**：需要连续性和一致性的交互设计
- **集成测试规划**：需要全局视角的测试策略

## 3. 使用技巧

### 提高效率的提示词模式

为获得最佳结果，使用以下提示词模式：

```
任务: [简洁描述任务类型]
背景: [提供必要上下文]
要求: [列出具体要求]
格式: [指明需要的输出格式]
```

### 命令行选项

```bash
# 基本使用
mcp-dev-assist "实现文件监控功能"

# 指定模型
mcp-dev-assist -m deepseek-coder "实现文件监控功能"

# 调整创造性
mcp-dev-assist -t 0.8 "生成测试用例"

# 启用缓存
mcp-dev-assist -c "解释Promise原理"
```

## 4. 工作流集成

### 与现有开发流程结合

1. **需求分析**：使用Claude讨论功能需求
2. **接口设计**：使用Claude设计核心接口
3. **代码实现**：使用DeepSeek生成实现代码
4. **测试编写**：使用DeepSeek生成单元测试
5. **文档生成**：使用DeepSeek创建文档

### 输出处理建议

- 审查所有生成的代码，不要盲目复制
- 根据项目编码规范调整生成的代码
- 确保生成的函数与已有代码风格一致
