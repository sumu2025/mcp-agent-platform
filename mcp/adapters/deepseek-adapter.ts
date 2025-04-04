/**
 * DeepSeekAdapter is a component that provides an interface for interacting with the DeepSeek service.
 * It handles API communication, error handling, and data transformation.
 */
class DeepSeekAdapter {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly defaultHeaders: Record<string, string>;

  /**
   * Creates an instance of DeepSeekAdapter.
   * @param {Object} config - Configuration object for the adapter
   * @param {string} config.baseUrl - The base URL for the DeepSeek API
   * @param {string} config.apiKey - The API key for authentication
   * @param {Record<string, string>} [config.defaultHeaders] - Optional default headers
   * @throws {Error} If required configuration parameters are missing
   */
  constructor(config: {
    baseUrl: string;
    apiKey: string;
    defaultHeaders?: Record<string, string>;
  }) {
    if (!config.baseUrl || !config.apiKey) {
      throw new Error('baseUrl and apiKey are required configuration parameters');
    }

    this.baseUrl = config.baseUrl;
    this.apiKey = config.apiKey;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${this.apiKey}`,
      ...(config.defaultHeaders || {}),
    };
  }

  /**
   * Makes a request to the DeepSeek API
   * @template T - The expected response type
   * @param {string} endpoint - The API endpoint to call
   * @param {RequestInit} [options] - Optional fetch options
   * @returns {Promise<T>} - The parsed response data
   * @throws {DeepSeekError} If the API request fails
   */
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      ...this.defaultHeaders,
      ...(options?.headers || {}),
    };

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const errorData = await this.parseErrorResponse(response);
        throw new DeepSeekError(
          errorData.message || 'API request failed',
          response.status,
          errorData
        );
      }

      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof DeepSeekError) {
        throw error;
      }
      throw new DeepSeekError(
        error instanceof Error ? error.message : 'Network request failed',
        0,
        { originalError: error }
      );
    }
  }

  /**
   * Parses error responses from the API
   * @param {Response} response - The failed response
   * @returns {Promise<Record<string, unknown>>} - The parsed error data
   */
  private async parseErrorResponse(
    response: Response
  ): Promise<Record<string, unknown>> {
    try {
      return (await response.json()) as Record<string, unknown>;
    } catch {
      return { message: response.statusText };
    }
  }

  /**
   * Executes a search query against the DeepSeek API
   * @template T - The expected response type
   * @param {string} query - The search query
   * @param {Record<string, unknown>} [params] - Optional query parameters
   * @returns {Promise<T>} - The search results
   */
  public async search<T>(
    query: string,
    params?: Record<string, unknown>
  ): Promise<T> {
    if (!query || typeof query !== 'string') {
      throw new Error('Query must be a non-empty string');
    }

    const endpoint = '/search';
    const options: RequestInit = {
      method: 'POST',
      body: JSON.stringify({ query, ...params }),
    };

    return this.request<T>(endpoint, options);
  }

  /**
   * Retrieves details for a specific item from the DeepSeek API
   * @template T - The expected response type
   * @param {string} id - The item ID
   * @returns {Promise<T>} - The item details
   */
  public async getItemDetails<T>(id: string): Promise<T> {
    if (!id || typeof id !== 'string') {
      throw new Error('ID must be a non-empty string');
    }

    const endpoint = `/items/${encodeURIComponent(id)}`;
    return this.request<T>(endpoint);
  }
}

/**
 * Custom error class for DeepSeek API errors
 */
class DeepSeekError extends Error {
  /**
   * Creates an instance of DeepSeekError
   * @param {string} message - The error message
   * @param {number} statusCode - The HTTP status code
   * @param {Record<string, unknown>} [details] - Additional error details
   */
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'DeepSeekError';
    Object.setPrototypeOf(this, DeepSeekError.prototype);
  }

  /**
   * Returns a string representation of the error
   * @returns {string} - The formatted error string
   */
  public toString(): string {
    return `${this.name}: ${this.message} (Status: ${this.statusCode})`;
  }
}

export { DeepSeekAdapter, DeepSeekError };