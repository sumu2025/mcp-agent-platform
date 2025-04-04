/**
 * DeepSeek真实API适配器
 */
const fetch = require("node-fetch");

class DeepSeekAdapter {
  /**
   * 创建DeepSeek适配器实例
   * @param {string} apiKey - DeepSeek API密钥
   * @param {Object} options - 配置选项
   */
  constructor(apiKey, options = {}) {
    if (!apiKey) throw new Error("API key is required");
    
    this.apiKey = apiKey;
    this.baseUrl = options.baseUrl || "https://api.deepseek.com";
    this.model = options.model || "deepseek-chat";
    this.maxRetries = options.maxRetries || 2;
    this.timeout = options.timeout || 30000;
  }

  /**
   * 生成文本响应
   * @param {string} prompt - 输入提示词
   * @param {Object} options - 可选参数
   * @returns {Promise<string>} 生成的文本
   */
  async generateText(prompt, options = {}) {
    const maxTokens = options.maxTokens || 1000;
    const temperature = options.temperature || 0.7;
    
    console.log("🔍 DeepSeek API调用中...");
    
    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model: this.model,
          messages: [{ role: "user", content: prompt }],
          max_tokens: maxTokens,
          temperature: temperature
        }),
        timeout: this.timeout
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`DeepSeek API error: ${response.status} ${JSON.stringify(errorData)}`);
      }

      const data = await response.json();
      return data.choices[0].message.content;
    } catch (error) {
      console.error("DeepSeek API调用失败:", error.message);
      throw error;
    }
  }

  /**
   * 检查模型是否可用
   * @returns {Promise<boolean>} 可用状态
   */
  async checkAvailability() {
    try {
      // 发送一个小型测试请求来验证API可用性
      const response = await fetch(`${this.baseUrl}/v1/models`, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${this.apiKey}`
        },
        timeout: 5000  // 短超时
      });
      
      return response.ok;
    } catch (error) {
      console.warn("DeepSeek API可用性检查失败:", error.message);
      return false;
    }
  }
}

module.exports = { DeepSeekAdapter };
