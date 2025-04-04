/**
 * DeepSeek API Client - 简化版
 * 
 * 用于开发阶段的快速DeepSeek API集成
 * 提供基本的文本生成功能，作为Claude API的替代
 */

const fetch = require('node-fetch');
const ResponseCache = require('./cache');

class DeepSeekClient {
  /**
   * 初始化DeepSeek客户端
   * @param {Object} config 配置对象
   * @param {string} config.apiKey DeepSeek API密钥
   * @param {string} config.baseUrl DeepSeek API基础URL
   * @param {string} config.model 默认使用的模型
   */
  constructor(config) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'https://api.deepseek.com/v1/chat/completions';
    this.model = config.model || 'deepseek-chat';
    this.defaultConfig = {
      temperature: 0.7,
      max_tokens: 1000
    };
    
    // 初始化缓存（如果启用）
    this.cache = config.useCache ? new ResponseCache(config.cacheOptions) : null;
  }

  /**
   * 生成文本响应
   * @param {string} prompt 提示文本
   * @param {Object} options 配置选项
   * @returns {Promise<string>} 生成的文本
   */
  async generateText(prompt, options = {}) {
    try {
      // 准备请求配置
      const requestOptions = {
        ...this.defaultConfig,
        ...options
      };

      // 准备请求体
      const requestBody = {
        model: options.model || this.model,
        messages: [
          { role: 'user', content: prompt }
        ],
        temperature: requestOptions.temperature,
        max_tokens: requestOptions.max_tokens
      };
      
      // 检查缓存
      if (this.cache && !options.skipCache) {
        const cacheKey = this.cache.generateKey(requestBody);
        const cachedResponse = this.cache.get(cacheKey);
        
        if (cachedResponse) {
          console.log('使用缓存的响应');
          return cachedResponse;
        }
      }

      // 发送请求
      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify(requestBody)
      });

      // 处理响应
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`DeepSeek API错误 (${response.status}): ${JSON.stringify(errorData)}`);
      }

      const data = await response.json();
      const result = data.choices[0].message.content;
      
      // 缓存响应
      if (this.cache && !options.skipCache) {
        const cacheKey = this.cache.generateKey(requestBody);
        this.cache.set(cacheKey, result);
      }
      
      return result;
    } catch (error) {
      console.error('DeepSeek API调用失败:', error);
      throw error;
    }
  }

  /**
   * 生成结构化响应
   * @param {string} prompt 提示文本
   * @param {Object} options 配置选项
   * @returns {Promise<Object>} 解析后的JSON对象
   */
  async generateStructured(prompt, options = {}) {
    try {
      // 向提示添加JSON格式要求
      const jsonPrompt = `${prompt}\n\n请以有效的JSON格式回答，确保输出可以被JSON.parse()正确解析。`;
      
      // 生成响应
      const rawResponse = await this.generateText(jsonPrompt, options);
      
      // 尝试解析JSON
      try {
        // 提取JSON部分（如果响应中包含其他文本）
        const jsonMatch = rawResponse.match(/```json\n([\s\S]*)\n```/) || 
                          rawResponse.match(/```\n([\s\S]*)\n```/) ||
                          [null, rawResponse];
        
        const jsonStr = jsonMatch[1].trim();
        return JSON.parse(jsonStr);
      } catch (parseError) {
        console.warn('无法解析JSON响应，返回原始文本', parseError);
        return { text: rawResponse, parseError: true };
      }
    } catch (error) {
      console.error('结构化生成失败:', error);
      throw error;
    }
  }

  /**
   * 检查API可用性
   * @returns {Promise<boolean>} API是否可用
   */
  async checkAvailability() {
    try {
      // 发送简单请求测试连接
      const result = await this.generateText('测试连接', { max_tokens: 5 });
      return result !== null && result !== undefined;
    } catch (error) {
      console.warn('DeepSeek API不可用:', error.message);
      return false;
    }
  }
}

module.exports = DeepSeekClient;
