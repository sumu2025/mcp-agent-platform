/**
 * DeepSeek适配器实用工具
 */

/**
 * 简单的提示词转换，调整为DeepSeek API偏好格式
 * @param {string} claudePrompt 为Claude格式化的提示
 * @returns {string} 为DeepSeek优化的提示
 */
function adaptPrompt(claudePrompt) {
  // 基本的提示词转换
  let adapted = claudePrompt
    // 移除Claude特定XML标签
    .replace(/<\/?thinking>|<\/?answer>/gi, '')
    // 调整分隔符风格
    .replace(/\n\n---\n\n/g, '\n\n----\n\n');
    
  // 增强指令清晰度
  if (!adapted.includes("请你") && !adapted.includes("我需要你")) {
    adapted = `请你回答以下问题或执行以下任务:\n\n${adapted}`;
  }
    
  // 特定格式处理
  adapted = adapted
    // JSON请求处理
    .replace(
      "<format>json</format>", 
      "请以有效的JSON格式回答，确保输出可以被JSON.parse()正确解析。"
    )
    // Markdown格式处理
    .replace(
      "<format>markdown</format>",
      "请使用Markdown格式回答，注意格式的正确性和可读性。"
    );
    
  return adapted;
}

/**
 * 将Claude风格的消息数组转换为DeepSeek消息格式
 * @param {Array} claudeMessages Claude风格的消息数组
 * @param {string} systemPrompt 可选的系统提示
 * @returns {Array} DeepSeek风格的消息数组
 */
function adaptMessages(claudeMessages, systemPrompt) {
  let messages = [];
  
  // 处理系统提示
  if (systemPrompt) {
    messages.push({
      role: "system",
      content: adaptPrompt(systemPrompt)
    });
  }
  
  // 转换消息数组
  claudeMessages.forEach(msg => {
    let role = msg.role;
    let content = msg.content;
    
    // 角色映射
    if (role === "human") role = "user";
    if (role === "assistant") role = "assistant";
    
    // 内容处理
    if (typeof content === "string") {
      content = adaptPrompt(content);
    } else if (Array.isArray(content)) {
      // 处理多模态内容
      content = adaptMultiModalContent(content);
    }
    
    messages.push({ role, content });
  });
  
  return messages;
}

/**
 * 处理多模态内容
 * @param {Array} content 多模态内容数组
 * @returns {string} 处理后的文本内容
 */
function adaptMultiModalContent(content) {
  // 默认实现: 提取文本部分
  let textContent = "";
  
  content.forEach(item => {
    if (item.type === "text") {
      textContent += item.text;
    } else if (item.type === "image") {
      // 添加图像引用
      textContent += "[图像内容]";
    }
  });
  
  return textContent;
}

module.exports = {
  adaptPrompt,
  adaptMessages,
  adaptMultiModalContent
};
