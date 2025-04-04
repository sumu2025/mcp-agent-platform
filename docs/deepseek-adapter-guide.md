# DeepSeek适配器开发指南

## 简介

`DeepSeekClient`是MCP智能体中台项目中的核心组件，用于与DeepSeek API通信，为系统提供文本生成和智能分析功能。本文档提供了适配器的详细使用方法，帮助开发者在项目中集成和使用DeepSeek功能。

## 快速开始

### 基本使用

```javascript
const { DeepSeekClient } = require('../mcp/adapters/deepseek');

// 创建客户端实例
const client = new DeepSeekClient({
  apiKey: 'your_api_key_here',
  model: 'deepseek-chat'  // 可选，默认为 'deepseek-chat'
});

// 生成文本
async function generateAnswer() {
  try {
    const response = await client.generateText("什么是向量数据库？");
    console.log(response);
  } catch (error) {
    console.error("生成失败:", error);
  }
}

generateAnswer();
```

### 使用提示词转换工具

```javascript
const { utils } = require('../mcp/adapters/deepseek');

// 将Claude格式的提示转换为DeepSeek格式
const claudePrompt = `<thinking>
我需要考虑数据库的类型和性能特点。
</thinking>

设计一个用户系统的数据库架构`;

const adaptedPrompt = utils.adaptPrompt(claudePrompt);
```

### 处理结构化输出

```javascript
const { DeepSeekClient } = require('../mcp/adapters/deepseek');

const client = new DeepSeekClient({
  apiKey: 'your_api_key_here'
});

async function getStructuredData() {
  try {
    const data = await client.generateStructured(
      "以JSON格式返回五个流行编程语言及其特点"
    );
    
    // data是解析后的JSON对象
    console.log(data.languages[0].name);
  } catch (error) {
    console.error("生成失败:", error);
  }
}

getStructuredData();
```

## API参考

### `DeepSeekClient`

主要客户端类，用于与DeepSeek API交互。

#### 构造函数

```javascript
new DeepSeekClient(config)
```

**参数:**
- `config.apiKey` (string): DeepSeek API密钥
- `config.baseUrl` (string): 可选，API基础URL
- `config.model` (string): 可选，默认使用的模型，默认为'deepseek-chat'

#### 方法

##### `generateText(prompt, options)`

生成文本响应。

**参数:**
- `prompt` (string): 提示文本
- `options` (object): 可选配置
  - `model`: 模型名称
  - `temperature`: 温度参数(0-1)
  - `max_tokens`: 最大生成token数

**返回:** Promise<string> - 生成的文本

##### `generateStructured(prompt, options)`

生成并解析JSON结构化数据。

**参数:**
- `prompt` (string): 提示文本
- `options` (object): 可选配置，同`generateText`

**返回:** Promise<object> - 解析后的JSON对象

##### `checkAvailability()`

检查API是否可用。

**返回:** Promise<boolean> - API是否可用

### `utils`

提供各种工具函数，用于处理提示词和消息格式。

#### `adaptPrompt(claudePrompt)`

将Claude格式的提示词转换为DeepSeek格式。

**参数:**
- `claudePrompt` (string): Claude格式的提示词

**返回:** string - 适配后的提示词

#### `adaptMessages(claudeMessages, systemPrompt)`

将Claude消息数组转换为DeepSeek消息格式。

**参数:**
- `claudeMessages` (array): Claude格式的消息数组
- `systemPrompt` (string): 可选，系统提示

**返回:** array - 适配后的消息数组

## 高级用法

### 自定义错误处理

```javascript
const { DeepSeekClient } = require('../mcp/adapters/deepseek');

const client = new DeepSeekClient({
  apiKey: 'your_api_key_here'
});

async function generateWithRetry(prompt, maxRetries = 3) {
  let retries = 0;
  
  while (retries < maxRetries) {
    try {
      return await client.generateText(prompt);
    } catch (error) {
      retries++;
      console.warn(`尝试 ${retries}/${maxRetries} 失败:`, error.message);
      
      if (retries >= maxRetries) {
        throw new Error(`在${maxRetries}次尝试后失败: ${error.message}`);
      }
      
      // 等待一段时间后重试
      await new Promise(r => setTimeout(r, 1000 * retries));
    }
  }
}
```

### 流式处理（计划中）

在后续版本中，我们将添加流式处理支持：

```javascript
// 未来版本功能，当前不可用
const stream = await client.streamText("生成一篇关于AI的文章");

for await (const chunk of stream) {
  process.stdout.write(chunk);
}
```

## 集成示例

### 与MCP上下文系统集成

```javascript
const { DeepSeekClient } = require('../mcp/adapters/deepseek');
const { ContextManager } = require('../mcp/core/context');

async function generateWithContext() {
  // 创建上下文管理器
  const contextManager = new ContextManager();
  
  // 加载项目上下文
  const contextPrompt = contextManager.buildContextPrompt();
  
  // 创建DeepSeek客户端
  const client = new DeepSeekClient({
    apiKey: 'your_api_key_here'
  });
  
  // 带上下文生成
  const prompt = `${contextPrompt}\n\n如何实现项目的下一个功能?`;
  const response = await client.generateText(prompt);
  
  return response;
}
```

## 最佳实践

1. **错误处理**：始终使用try/catch处理API调用异常
2. **提示词优化**：使用utils.adaptPrompt优化提示词格式
3. **资源管理**：对长时间运行的应用，定期检查API可用性
4. **安全性**：不要在客户端代码中硬编码API密钥
5. **性能优化**：对频繁使用的提示实现本地缓存

## 扩展计划

DeepSeek适配器将在未来版本中添加以下功能：

1. 流式输出支持
2. 更多模型支持
3. 完整的Claude API兼容层
4. 多模态内容处理
5. 自动模型切换机制

## 故障排除

常见问题及解决方案：

1. **API连接错误**：检查网络连接和防火墙设置
2. **认证失败**：确认API密钥正确且未过期
3. **响应解析错误**：检查生成的内容是否符合预期格式
4. **配额限制**：检查API使用配额是否已达上限

---

如有问题或需要进一步帮助，请参考DeepSeek官方文档或联系项目维护者。
