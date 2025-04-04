export interface IModelAdapter {
  generateText(prompt: string, options?: any): Promise<string>;
  checkAvailability(): Promise<boolean>;
}
