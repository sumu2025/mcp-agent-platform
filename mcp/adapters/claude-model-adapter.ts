/**
 * Interface defining the configuration options for ClaudeModelAdapter.
 */
interface ClaudeModelAdapterConfig {
  apiKey: string;
  modelVersion?: string;
  maxRetries?: number;
  timeout?: number;
}

/**
 * Interface defining the response structure from Claude model.
 */
interface ClaudeModelResponse {
  success: boolean;
  data?: any;
  error?: {
    message: string;
    code?: number;
  };
}

/**
 * Adapter class for interacting with Claude AI models.
 * Provides a clean interface for sending prompts and receiving responses.
 */
class ClaudeModelAdapter {
  private readonly apiKey: string;
  private readonly modelVersion: string;
  private readonly maxRetries: number;
  private readonly timeout: number;

  /**
   * Creates a new instance of ClaudeModelAdapter.
   * @param {ClaudeModelAdapterConfig} config - Configuration object for the adapter.
   * @throws {Error} If required configuration is missing or invalid.
   */
  constructor(config: ClaudeModelAdapterConfig) {
    if (!config.apiKey) {
      throw new Error('API key is required for ClaudeModelAdapter');
    }

    this.apiKey = config.apiKey;
    this.modelVersion = config.modelVersion || 'latest';
    this.maxRetries = config.maxRetries || 3;
    this.timeout = config.timeout || 10000;
  }

  /**
   * Sends a prompt to the Claude model and returns the response.
   * @param {string} prompt - The input prompt to send to the model.
   * @param {Record<string, any>} [options] - Additional options for the request.
   * @returns {Promise<ClaudeModelResponse>} The response from the model.
   */
  async sendPrompt(
    prompt: string,
    options?: Record<string, any>
  ): Promise<ClaudeModelResponse> {
    if (!prompt || typeof prompt !== 'string') {
      throw new Error('Prompt must be a non-empty string');
    }

    try {
      // In a real implementation, this would make an actual API call
      // For demonstration, we're returning a mock response
      return await this.mockApiCall(prompt, options);
    } catch (error) {
      console.error('Error in ClaudeModelAdapter:', error);
      return {
        success: false,
        error: {
          message: error instanceof Error ? error.message : 'Unknown error occurred',
          code: 500,
        },
      };
    }
  }

  /**
   * Mock API call implementation for demonstration purposes.
   * In a real implementation, this would be replaced with actual HTTP calls.
   * @private
   */
  private async mockApiCall(
    prompt: string,
    options?: Record<string, any>
  ): Promise<ClaudeModelResponse> {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          success: true,
          data: {
            prompt,
            options,
            response: `Mock response for prompt: ${prompt.substring(0, 50)}...`,
            modelVersion: this.modelVersion,
          },
        });
      }, 500); // Simulate network latency
    });
  }

  /**
   * Validates the response from the Claude model.
   * @private
   */
  private validateResponse(response: any): response is ClaudeModelResponse {
    return (
      typeof response === 'object' &&
      response !== null &&
      'success' in response &&
      typeof response.success === 'boolean'
    );
  }
}

export { ClaudeModelAdapter, type ClaudeModelAdapterConfig, type ClaudeModelResponse };