/**
 * ClaudeAdapter is a component that provides an interface to interact with the Claude API.
 * It handles API communication, error handling, and response processing.
 */
export class ClaudeAdapter {
  private readonly apiKey: string;
  private readonly baseUrl: string;

  /**
   * Creates an instance of ClaudeAdapter.
   * @param {string} apiKey - The API key for authentication with Claude service
   * @param {string} [baseUrl='https://api.claude.ai'] - The base URL for the Claude API
   * @throws {Error} If apiKey is not provided or is invalid
   */
  constructor(apiKey: string, baseUrl: string = 'https://api.claude.ai') {
    if (!apiKey || typeof apiKey !== 'string') {
      throw new Error('Invalid API key. A valid string API key is required.');
    }

    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  /**
   * Sends a prompt to Claude API and returns the response.
   * @param {string} prompt - The input prompt to send to Claude
   * @param {object} [options] - Additional options for the request
   * @param {number} [options.maxTokens=100] - Maximum number of tokens in the response
   * @param {number} [options.temperature=0.7] - Controls randomness in the response
   * @returns {Promise<string>} The generated response from Claude
   * @throws {Error} If the prompt is invalid or API request fails
   */
  async sendPrompt(
    prompt: string,
    options: { maxTokens?: number; temperature?: number } = {}
  ): Promise<string> {
    if (!prompt || typeof prompt !== 'string') {
      throw new Error('Prompt must be a non-empty string');
    }

    const { maxTokens = 100, temperature = 0.7 } = options;

    try {
      const response = await fetch(`${this.baseUrl}/v1/complete`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          prompt,
          max_tokens: maxTokens,
          temperature,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          `API request failed with status ${response.status}: ${errorData.message || 'Unknown error'}`
        );
      }

      const data = await response.json();
      return data.completion;
    } catch (error) {
      if (error instanceof Error) {
        throw new Error(`Failed to send prompt to Claude: ${error.message}`);
      }
      throw new Error('Failed to send prompt to Claude: Unknown error occurred');
    }
  }

  /**
   * Validates the API key by making a test request to the Claude API.
   * @returns {Promise<boolean>} True if the API key is valid, false otherwise
   */
  async validateApiKey(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/validate`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        },
      });

      return response.ok;
    } catch {
      return false;
    }
  }
}