/**
 * Claude真实API适配器
 */
const fetch = require("node-fetch");

class ClaudeAdapter {
  /**
   * 创建Claude适配器实例
   * @param {string} apiKey - Claude API密钥
   * @param {Object} options - 配置选项
   */
  constructor(apiKey, options = {}) {
    if (!apiKey) throw new Error("API key is required");
    
    this.apiKey = apiKey;
    this.baseUrl = options.baseUrl || "https://api.anthropic.com";
    this.modelVersion = options.modelVersion || "claude-3-sonnet-20240229";
    this.maxRetries = options.maxRetries || 2;
    this.timeout = options.timeout || 30000;
    this.simulateFailure = false;
  }

  /**
   * 生成文本响应
   * @param {string} prompt - 输入提示词
   * @param {Object} options - 可选参数
   * @returns {Promise<string>} 生成的文本
   */
  async generateText(prompt, options = {}) {
    if (this.simulateFailure) {
      throw new Error("Claude service unavailable (simulated)");
    }

    const maxTokens = options.maxTokens || 1000;
    const temperature = options.temperature || 0.7;
    
    console.log("🧠 Claude API调用中...");
    
    try {
      const response = await fetch(`${this.baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": this.apiKey,
          "anthropic-version": "2023-06-01"
        },
        body: JSON.stringify({
          model: this.modelVersion,
          messages: [{ role: "user", content: prompt }],
          max_tokens: maxTokens,
          temperature: temperature
        }),
        timeout: this.timeout
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Claude API error: ${response.status} ${JSON.stringify(errorData)}`);
      }

      const data = await response.json();
      return data.content[0].text;
    } catch (error) {
      console.error("Claude API调用失败:", error.message);
      throw error;
    }
  }

  /**
   * 检查模型是否可用
   * @returns {Promise<boolean>} 可用状态
   */
  async checkAvailability() {
    if (this.simulateFailure) return false;
    
    try {
      // 发送一个小型测试请求来验证API可用性
      const response = await fetch(`${this.baseUrl}/v1/models`, {
        method: "GET",
        headers: {
          "x-api-key": this.apiKey,
          "anthropic-version": "2023-06-01"
        },
        timeout: 5000  // 短超时
      });
      
      return response.ok;
    } catch (error) {
      console.warn("Claude API可用性检查失败:", error.message);
      return false;
    }
  }

  /**
   * 设置故障模拟
   * @param {boolean} simulate - 是否模拟故障
   */
  setSimulateFailure(simulate) {
    this.simulateFailure = simulate;
  }
}

module.exports = { ClaudeAdapter };
