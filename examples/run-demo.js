require("dotenv").config();
const { FallbackDemoApp } = require("./fallback-demo-app");

async function main() {
  console.log("正在启动MCP降级演示...");
  
  const claudeApiKey = process.env.CLAUDE_API_KEY || "mock_claude_key";
  const deepSeekApiKey = process.env.DEEPSEEK_API_KEY || "mock_deepseek_key";
  
  const app = new FallbackDemoApp(claudeApiKey, deepSeekApiKey);
  await app.runDemo();
}

main().catch(error => {
  console.error("演示执行失败:", error);
  process.exit(1);
});
