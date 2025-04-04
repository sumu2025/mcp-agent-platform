require("dotenv").config();
const { ClaudeAdapter } = require("../mcp/adapters/claude-adapter-fixed");
const { DeepSeekAdapter } = require("../mcp/adapters/deepseek-adapter-real");

async function main() {
  // 获取API密钥
  const claudeApiKey = process.env.CLAUDE_API_KEY;
  const deepSeekApiKey = process.env.DEEPSEEK_API_KEY;
  
  console.log("===== API连接测试 =====");
  
  // 测试Claude
  console.log("测试Claude API连接...");
  const claudeAdapter = new ClaudeAdapter(claudeApiKey);
  try {
    const isClaudeAvailable = await claudeAdapter.checkAvailability();
    console.log(`Claude可用性: ${isClaudeAvailable ? "可用" : "不可用"}`);
    
    if (isClaudeAvailable) {
      console.log("测试Claude文本生成...");
      const claudeResponse = await claudeAdapter.generateText(
        "简短介绍一下你自己", 
        { maxTokens: 100 }
      );
      console.log("Claude响应:", claudeResponse);
    }
  } catch (error) {
    console.error("Claude API测试失败:", error);
  }
  
  // 测试DeepSeek
  console.log("测试DeepSeek API连接...");
  const deepSeekAdapter = new DeepSeekAdapter(deepSeekApiKey);
  try {
    const isDeepSeekAvailable = await deepSeekAdapter.checkAvailability();
    console.log(`DeepSeek可用性: ${isDeepSeekAvailable ? "可用" : "不可用"}`);
    
    if (isDeepSeekAvailable) {
      console.log("测试DeepSeek文本生成...");
      const deepSeekResponse = await deepSeekAdapter.generateText(
        "简短介绍一下你自己", 
        { maxTokens: 100 }
      );
      console.log("DeepSeek响应:", deepSeekResponse);
    }
  } catch (error) {
    console.error("DeepSeek API测试失败:", error);
  }
  
  console.log("===== 测试完成 =====");
}

main().catch(console.error);
