import { IModelAdapter } from "./i-model-adapter";

export class ModelManager {
  private primaryAdapter: IModelAdapter;
  private fallbackAdapter: IModelAdapter;
  private currentAdapter: IModelAdapter;
  
  constructor(primaryAdapter: IModelAdapter, fallbackAdapter: IModelAdapter) {
    this.primaryAdapter = primaryAdapter;
    this.fallbackAdapter = fallbackAdapter;
    this.currentAdapter = primaryAdapter;
  }
  
  async processPrompt(prompt: string): Promise<string> {
    try {
      // 检查主要适配器是否可用
      if (this.currentAdapter === this.primaryAdapter) {
        const isAvailable = await this.primaryAdapter.checkAvailability();
        if (!isAvailable) {
          console.log("主要模型不可用，切换到备用模型");
          this.currentAdapter = this.fallbackAdapter;
        }
      }
      
      // 使用当前适配器处理请求
      return await this.currentAdapter.generateText(prompt);
    } catch (error) {
      // 如果使用主要适配器失败，切换到备用适配器
      if (this.currentAdapter === this.primaryAdapter) {
        console.log("主要模型处理失败，切换到备用模型");
        this.currentAdapter = this.fallbackAdapter;
        return await this.currentAdapter.generateText(prompt);
      }
      
      // 如果备用适配器也失败，抛出错误
      throw error;
    }
  }
  
  getCurrentModelName(): string {
    return this.currentAdapter === this.primaryAdapter ? "Claude" : "DeepSeek";
  }
  
  resetToDefault(): void {
    this.currentAdapter = this.primaryAdapter;
  }
}
