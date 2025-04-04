# DeepSeek开发助手设置指南

本文档指导您设置和使用DeepSeek API作为MCP项目的开发助手，以提高开发效率并减轻Claude API的token负担。

## 获取DeepSeek API密钥

1. 访问[DeepSeek AI开发者平台](https://platform.deepseek.com/)
2. 注册/登录您的账户
3. 导航至"API密钥"或"开发者设置"部分
4. 创建新的API密钥并保存（不要共享或泄露此密钥）

## 安装开发助手工具

### 方法1：本地安装（推荐）

```bash
# 在项目根目录执行
cd /path/to/mcp_agent_platform
npm install

# 设置执行权限
chmod +x ./bin/mcp-dev-assist.js

# 创建全局命令链接
npm link
```

### 方法2：直接使用

```bash
# 在项目根目录执行
cd /path/to/mcp_agent_platform

# 通过npm脚本运行
npm run dev-assist "你的开发问题或请求"
```

## 首次使用

首次使用时，工具会提示您输入DeepSeek API密钥：

```bash
mcp-dev-assist "设计一个文件监控系统"
```

输入API密钥后，它将被保存到`~/.mcp/config.json`，后续使用将自动加载。

## 使用场景

### 适合使用DeepSeek的场景

- 代码生成和补全
- 常规开发问题
- 文档编写
- 简单设计讨论

示例：
```bash
mcp-dev-assist "为Obsidian文件监控器实现一个事件处理器"
```

### 保留给Claude的场景

- 复杂架构决策
- 关键设计评审
- 项目整体规划
- 多轮深入讨论

## 使用技巧

1. **具体问题**：明确描述您的需求和上下文
2. **包含示例**：提供相关代码段或要求的示例
3. **指定格式**：如果需要特定格式，请明确说明
4. **引用文件**：可以引用项目中的文件路径

## 故障排除

如果遇到问题：

1. 检查API密钥是否正确
2. 确保网络连接正常
3. 查看日志输出
4. 重置配置：删除`~/.mcp/config.json`文件

## DeepSeek vs Claude 使用指南

| 任务类型 | 首选工具 | 备注 |
|---------|---------|------|
| 生成代码片段 | DeepSeek | 更高频率限制，适合多次迭代 |
| 代码阅读和解释 | DeepSeek | 适合日常开发辅助 |
| 文档生成 | DeepSeek | 适合标准文档 |
| 复杂架构决策 | Claude | 需要深入理解和推理 |
| 多轮对话 | Claude | 上下文理解更好 |
| 系统设计 | Claude | 先使用Claude讨论，再用DeepSeek实现细节 |
