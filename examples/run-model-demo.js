// 模型降级演示运行脚本
require("dotenv").config();
const { ModelFallbackDemo } = require("./model-fallback-demo");

async function runDemo() {
  // 从环境变量获取API密钥
  const claudeApiKey = process.env.CLAUDE_API_KEY || "simulated_claude_key";
  const deepSeekApiKey = process.env.DEEPSEEK_API_KEY || "simulated_deepseek_key";
  
  console.log("=== MCP模型降级演示 ===");
  
  const demo = new ModelFallbackDemo(claudeApiKey, deepSeekApiKey);
  await demo.runDemo();
  
  console.log("=== 演示完成 ===");
}

runDemo().catch(error => {
  console.error("演示出错:", error);
});

