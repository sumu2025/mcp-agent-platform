/**
 * DeepSeek代码生成器
 * 专门用于生成高质量的代码和组件
 * 
 * 这个模块扩展了DeepSeek适配器，提供专业代码生成功能
 */

const DeepSeekClient = require('./client');
const utils = require('./utils');

class CodeGenerator {
  /**
   * 初始化代码生成器
   * @param {Object} config 配置对象
   * @param {string} config.apiKey DeepSeek API密钥
   * @param {string} config.model 使用的模型，默认为deepseek-coder
   * @param {Object} config.defaults 默认生成参数
   */
  constructor(config) {
    this.client = new DeepSeekClient({
      apiKey: config.apiKey,
      baseUrl: config.baseUrl,
      model: config.model || 'deepseek-coder'
    });

    this.defaults = {
      temperature: 0.2,
      max_tokens: 2000,
      language: 'typescript',
      ...config.defaults
    };
  }

  /**
   * 生成组件代码
   * @param {Object} spec 组件规格
   * @param {string} spec.name 组件名称
   * @param {string} spec.description 组件描述
   * @param {string} spec.type 组件类型（class, function, module等）
   * @param {Object} options 生成选项
   * @returns {Promise<string>} 生成的代码
   */
  async generateComponent(spec, options = {}) {
    const { name, description, type, interfaces = [], methods = [] } = spec;
    const language = options.language || this.defaults.language;
    
    // 构建详细提示
    const prompt = `
请生成一个高质量的${type === 'class' ? '类' : '组件'}，符合以下规格：

名称: ${name}
描述: ${description}
类型: ${type}
${interfaces.length > 0 ? `实现接口: ${interfaces.join(', ')}` : ''}
${methods.length > 0 ? `主要方法:\n${methods.map(m => `- ${m}`).join('\n')}` : ''}

请遵循以下要求：
1. 使用${language}编写
2. 添加完整的JSDoc注释
3. 实现错误处理和类型检查
4. 代码应符合现代${language}最佳实践
5. 代码应该模块化和可测试
6. 不要包含任何额外的说明，直接提供代码

代码:`;

    // 生成代码
    const response = await this.client.generateText(prompt, {
      temperature: options.temperature || this.defaults.temperature,
      max_tokens: options.max_tokens || this.defaults.max_tokens
    });

    // 提取代码部分（如果有多余内容）
    return this._extractCode(response, language);
  }

  /**
   * 生成API模块代码
   * @param {Object} spec API规格
   * @param {string} spec.name API名称
   * @param {string} spec.description API描述
   * @param {Array} spec.endpoints API端点列表
   * @param {Object} options 生成选项
   * @returns {Promise<string>} 生成的代码
   */
  async generateApi(spec, options = {}) {
    const { name, description, endpoints = [] } = spec;
    const language = options.language || this.defaults.language;
    
    // 端点详情格式化
    const endpointsText = endpoints.map(ep => 
      `- ${ep.method} ${ep.path}: ${ep.description}`
    ).join('\n');

    // 构建详细提示
    const prompt = `
请生成一个完整的API模块，符合以下规格：

名称: ${name}
描述: ${description}
API端点:
${endpointsText}

请遵循以下要求：
1. 使用${language}编写
2. 使用现代化的HTTP客户端实现
3. 包含请求验证和错误处理
4. 添加完整的JSDoc注释
5. 实现所有列出的端点
6. 遵循RESTful设计原则
7. 不要包含任何额外的说明，直接提供代码

代码:`;

    // 生成代码
    const response = await this.client.generateText(prompt, {
      temperature: options.temperature || this.defaults.temperature,
      max_tokens: options.max_tokens || this.defaults.max_tokens
    });

    // 提取代码部分
    return this._extractCode(response, language);
  }

  /**
   * 生成单元测试代码
   * @param {Object} spec 测试规格
   * @param {string} spec.targetFile 目标文件路径
   * @param {string} spec.targetCode 目标代码（可选，如果有文件路径可不提供）
   * @param {Object} options 生成选项
   * @returns {Promise<string>} 生成的测试代码
   */
  async generateTests(spec, options = {}) {
    const { targetFile, targetCode, testFramework = 'jest' } = spec;
    const language = options.language || this.defaults.language;
    
    // 构建详细提示
    let prompt = `
请为以下代码生成详尽的单元测试：

目标文件: ${targetFile}
测试框架: ${testFramework}

${targetCode ? '目标代码:\n```\n' + targetCode + '\n```\n' : ''}

请遵循以下要求：
1. 使用${testFramework}框架编写测试
2. 测试应该涵盖所有公共函数和方法
3. 包含正常场景和错误场景测试
4. 使用模拟(mock)处理外部依赖
5. 测试应该独立且可重复运行
6. 添加清晰的测试描述
7. 不要包含任何额外的说明，直接提供代码

测试代码:`;

    // 生成代码
    const response = await this.client.generateText(prompt, {
      temperature: options.temperature || this.defaults.temperature,
      max_tokens: options.max_tokens || this.defaults.max_tokens
    });

    // 提取代码部分
    return this._extractCode(response, language);
  }

  /**
   * 生成适配器代码
   * @param {Object} spec 适配器规格
   * @param {string} spec.sourceName 源系统名称
   * @param {string} spec.targetName 目标系统名称
   * @param {Object} spec.sourceInterface 源接口定义
   * @param {Object} spec.targetInterface 目标接口定义
   * @param {Object} options 生成选项
   * @returns {Promise<string>} 生成的适配器代码
   */
  async generateAdapter(spec, options = {}) {
    const { sourceName, targetName, sourceInterface, targetInterface } = spec;
    const language = options.language || this.defaults.language;
    
    // 构建详细提示
    const prompt = `
请生成一个适配器，将${sourceName}连接到${targetName}，符合以下规格：

源系统: ${sourceName}
目标系统: ${targetName}

源接口:
\`\`\`
${JSON.stringify(sourceInterface, null, 2)}
\`\`\`

目标接口:
\`\`\`
${JSON.stringify(targetInterface, null, 2)}
\`\`\`

请遵循以下要求：
1. 使用${language}编写
2. 实现适配器设计模式
3. 处理所有接口差异和转换
4. 添加完整的错误处理
5. 代码应该可测试和可维护
6. 添加详细的JSDoc注释
7. 不要包含任何额外的说明，直接提供代码

适配器代码:`;

    // 生成代码
    const response = await this.client.generateText(prompt, {
      temperature: options.temperature || this.defaults.temperature,
      max_tokens: options.max_tokens || this.defaults.max_tokens
    });

    // 提取代码部分
    return this._extractCode(response, language);
  }

  /**
   * 从文本中提取代码部分
   * @private
   * @param {string} text 包含代码的文本
   * @param {string} language 编程语言
   * @returns {string} 提取出的代码
   */
  _extractCode(text, language) {
    // 尝试查找代码块
    const codeBlockRegex = new RegExp(`\`\`\`(?:${language})?\\s*([\\s\\S]*?)\\s*\`\`\``, 'i');
    const match = text.match(codeBlockRegex);
    
    if (match && match[1]) {
      return match[1].trim();
    }
    
    // 如果没有找到代码块，假设整个响应就是代码
    // 但先检查一下是否有一些常见的前导文本
    const lines = text.split('\n');
    let startLine = 0;
    
    // 跳过可能的前导文本，如"以下是代码："等
    for (let i = 0; i < Math.min(5, lines.length); i++) {
      if (lines[i].match(/以下|下面|这里|代码|实现|生成/)) {
        startLine = i + 1;
      }
    }
    
    return lines.slice(startLine).join('\n').trim();
  }
}

module.exports = CodeGenerator;
