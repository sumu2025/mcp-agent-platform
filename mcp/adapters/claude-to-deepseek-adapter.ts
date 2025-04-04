/**
 * Adapter to convert Claude API calls to DeepSeek API format
 */
class ClaudeToDeepSeekAdapter {
  private deepSeekClient: any;

  /**
   * Create a new adapter instance
   * @param {any} deepSeekClient - Initialized DeepSeek client instance
   */
  constructor(deepSeekClient: any) {
    this.deepSeekClient = deepSeekClient;
  }

  /**
   * Convert Claude text generation parameters to DeepSeek format
   * @param {string} prompt - Input prompt
   * @param {number} maxTokens - Maximum tokens to generate
   * @param {number} temperature - Sampling temperature
   * @param {number} topP - Top-p sampling value
   * @param {string[]} stopSequences - Sequences to stop generation
   * @returns {Promise<string>} Generated text
   * @throws {Error} If API call fails
   */
  async generateText(
    prompt: string,
    maxTokens: number,
    temperature: number,
    topP?: number,
    stopSequences?: string[]
  ): Promise<string> {
    try {
      const options = {
        max_tokens: maxTokens,
        temperature: temperature,
        ...(topP && { top_p: topP }),
        ...(stopSequences && { stop_sequences: stopSequences }),
      };

      return await this.deepSeekClient.generateText(prompt, options);
    } catch (error) {
      throw new Error(`Failed to generate text: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Convert Claude structured generation parameters to DeepSeek format
   * @param {string} prompt - Input prompt
   * @param {object} responseFormat - Desired response format
   * @param {number} maxTokens - Maximum tokens to generate
   * @returns {Promise<any>} Structured response
   * @throws {Error} If API call fails
   */
  async generateStructured(
    prompt: string,
    responseFormat: object,
    maxTokens: number
  ): Promise<any> {
    try {
      const options = {
        response_format: responseFormat,
        max_tokens: maxTokens,
      };

      return await this.deepSeekClient.generateStructured(prompt, options);
    } catch (error) {
      throw new Error(
        `Failed to generate structured response: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }
}

export default ClaudeToDeepSeekAdapter;