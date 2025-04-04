/**
 * MCP提示词模板库
 * 
 * 提供各种专业任务的提示词模板，用于生成高质量代码和内容
 */

/**
 * 代码生成相关模板
 */
const codeTemplates = {
  /**
   * 组件生成模板
   * 占位符: {name}, {description}, {type}, {language}, {interfaces}, {methods}
   */
  component: `
请生成一个高质量的{type}，符合以下规格：

名称: {name}
描述: {description}
类型: {type}
{interfaces}
{methods}

请遵循以下要求：
1. 使用{language}编写
2. 添加完整的JSDoc注释
3. 实现错误处理和类型检查
4. 代码应符合现代{language}最佳实践
5. 代码应该模块化和可测试
6. 确保所有公共接口都有类型定义
7. 不要包含任何额外的说明，直接提供代码

代码:`,

  /**
   * API模块生成模板
   * 占位符: {name}, {description}, {endpoints}, {language}
   */
  api: `
请生成一个完整的API模块，符合以下规格：

名称: {name}
描述: {description}
API端点:
{endpoints}

请遵循以下要求：
1. 使用{language}编写
2. 使用现代化的HTTP客户端实现
3. 包含请求验证和错误处理
4. 添加完整的JSDoc注释
5. 实现所有列出的端点
6. 遵循RESTful设计原则
7. 不要包含任何额外的说明，直接提供代码

代码:`,

  /**
   * 测试生成模板
   * 占位符: {targetFile}, {targetCode}, {framework}, {language}
   */
  test: `
请为以下代码生成详尽的单元测试：

目标文件: {targetFile}
测试框架: {framework}

目标代码:
\`\`\`
{targetCode}
\`\`\`

请遵循以下要求：
1. 使用{framework}框架编写测试
2. 测试应该涵盖所有公共函数和方法
3. 包含正常场景和错误场景测试
4. 使用模拟(mock)处理外部依赖
5. 测试应该独立且可重复运行
6. 添加清晰的测试描述
7. 不要包含任何额外的说明，直接提供代码

测试代码:`,

  /**
   * 适配器生成模板
   * 占位符: {sourceName}, {targetName}, {sourceInterface}, {targetInterface}, {language}
   */
  adapter: `
请生成一个适配器，将{sourceName}连接到{targetName}，符合以下规格：

源系统: {sourceName}
目标系统: {targetName}

源接口:
\`\`\`
{sourceInterface}
\`\`\`

目标接口:
\`\`\`
{targetInterface}
\`\`\`

请遵循以下要求：
1. 使用{language}编写
2. 实现适配器设计模式
3. 处理所有接口差异和转换
4. 添加完整的错误处理
5. 代码应该可测试和可维护
6. 添加详细的JSDoc注释
7. 不要包含任何额外的说明，直接提供代码

适配器代码:`,

  /**
   * 实用函数生成模板
   * 占位符: {name}, {description}, {functions}, {language}
   */
  utils: `
请生成一个实用工具模块，符合以下规格：

名称: {name}
描述: {description}
提供的功能:
{functions}

请遵循以下要求：
1. 使用{language}编写
2. 实现所有列出的功能
3. 添加详细的JSDoc注释
4. 每个功能应该是纯函数，易于测试
5. 处理边缘情况和错误
6. 设计合理的参数和返回类型
7. 不要包含任何额外的说明，直接提供代码

工具模块代码:`
};

/**
 * 文档生成相关模板
 */
const docTemplates = {
  /**
   * 模块文档生成模板
   * 占位符: {name}, {description}, {methods}, {examples}
   */
  moduleDoc: `
# {name}

{description}

## 安装

\`\`\`bash
npm install {name}
\`\`\`

## 使用方法

\`\`\`javascript
const {name} = require('{name}');

// 使用示例
{examples}
\`\`\`

## API参考

{methods}

## 许可证

MIT
`,

  /**
   * API文档生成模板
   * 占位符: {name}, {description}, {endpoints}
   */
  apiDoc: `
# {name} API参考

{description}

## 端点

{endpoints}

## 认证

所有API请求都需要一个有效的API密钥，通过\`Authorization\`请求头传递：

\`\`\`
Authorization: Bearer YOUR_API_KEY
\`\`\`

## 错误处理

API使用标准HTTP状态码指示请求状态。一般来说：

- 2xx: 成功
- 4xx: 客户端错误
- 5xx: 服务器错误

错误响应将包含以下JSON格式：

\`\`\`json
{
  "error": {
    "code": "error_code",
    "message": "Human-readable error message"
  }
}
\`\`\`

## 限流

API请求限制为每分钟60个请求。超过此限制将返回429状态码。
`
};

/**
 * 架构设计相关模板
 */
const architectureTemplates = {
  /**
   * 组件设计模板
   * 占位符: {name}, {description}, {requirements}
   */
  componentDesign: `
# {name} 组件设计

## 概述

{description}

## 需求

{requirements}

## 组件架构

[组件架构描述]

## 公共接口

[接口定义]

## 数据流

[数据流描述]

## 依赖关系

[依赖关系描述]

## 实现注意事项

[实现注意事项]

## 扩展点

[扩展点描述]
`,

  /**
   * 系统架构模板
   * 占位符: {name}, {description}, {requirements}, {components}
   */
  systemArchitecture: `
# {name} 系统架构设计

## 概述

{description}

## 需求

{requirements}

## 架构决策

[架构决策记录]

## 组件详情

{components}

## 接口详情

[接口详情]

## 数据流

[数据流描述]

## 部署考虑

[部署考虑]

## 扩展策略

[扩展策略]
`
};

// 导出所有模板
module.exports = {
  codeTemplates,
  docTemplates,
  architectureTemplates,
  
  /**
   * 填充模板函数
   * @param {string} template 模板字符串
   * @param {Object} values 要填充的值
   * @returns {string} 填充后的字符串
   */
  fillTemplate(template, values) {
    let result = template;
    for (const [key, value] of Object.entries(values)) {
      result = result.replace(new RegExp(`\\{${key}\\}`, 'g'), value);
    }
    return result;
  }
};
