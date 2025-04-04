import { IModelAdapter } from "../core/i-model-adapter";

export class ClaudeAdapter implements IModelAdapter {
  private isAvailable: boolean = true;
  private simulateFailure: boolean = false;

  constructor(private readonly apiKey: string) {
    if (!apiKey) throw new Error("API key is required");
  }

  async generateText(prompt: string, options?: any): Promise<string> {
    if (this.simulateFailure) {
      throw new Error("Claude service unavailable (simulated)");
    }

    // 模拟API调用
    console.log("🧠 Claude正在处理...");
    await new Promise(resolve => setTimeout(resolve, 1000));
    return `[Claude] 响应: ${prompt.substring(0, 20)}...`;
  }

  async checkAvailability(): Promise<boolean> {
    if (this.simulateFailure) return false;
    return this.isAvailable;
  }

  setSimulateFailure(simulate: boolean): void {
    this.simulateFailure = simulate;
  }
}
