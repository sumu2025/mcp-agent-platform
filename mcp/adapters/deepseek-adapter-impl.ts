import { IModelAdapter } from "../core/i-model-adapter";

export class DeepSeekAdapter implements IModelAdapter {
  private isAvailable: boolean = true;
  
  constructor(private readonly apiKey: string) {
    if (!apiKey) throw new Error("API key is required");
  }

  async generateText(prompt: string, options?: any): Promise<string> {
    // 模拟API调用
    console.log("🔍 DeepSeek正在处理...");
    await new Promise(resolve => setTimeout(resolve, 800));
    return `[DeepSeek] 响应: ${prompt.substring(0, 20)}...`;
  }

  async checkAvailability(): Promise<boolean> {
    return this.isAvailable;
  }
}
