/**
 * DeepSeekModelAdapter is a component that provides an interface to interact with DeepSeek AI models.
 * It handles model initialization, data processing, and error management in a standardized way.
 * 
 * @class DeepSeekModelAdapter
 * @template TInput The input type expected by the model
 * @template TOutput The output type returned by the model
 */
export class DeepSeekModelAdapter<TInput, TOutput> {
  private modelInitialized: boolean = false;

  /**
   * Creates an instance of DeepSeekModelAdapter.
   * @param {string} modelName - The name/identifier of the DeepSeek model to use
   * @param {(input: TInput) => Promise<TOutput>} modelExecutor - The function that executes the model
   * @throws {Error} If modelName is empty or modelExecutor is not a function
   */
  constructor(
    private readonly modelName: string,
    private readonly modelExecutor: (input: TInput) => Promise<TOutput>
  ) {
    if (!modelName || typeof modelName !== 'string') {
      throw new Error('Invalid modelName provided');
    }

    if (typeof modelExecutor !== 'function') {
      throw new Error('modelExecutor must be a function');
    }
  }

  /**
   * Initializes the model adapter.
   * @async
   * @returns {Promise<void>}
   * @throws {Error} If initialization fails
   */
  public async initialize(): Promise<void> {
    try {
      // Placeholder for actual initialization logic
      this.modelInitialized = true;
    } catch (error) {
      throw new Error(`Failed to initialize model adapter: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Executes the model with the provided input.
   * @async
   * @param {TInput} input - The input data for the model
   * @returns {Promise<TOutput>} The model's output
   * @throws {Error} If the model is not initialized or execution fails
   */
  public async execute(input: TInput): Promise<TOutput> {
    if (!this.modelInitialized) {
      throw new Error('Model adapter is not initialized. Call initialize() first.');
    }

    try {
      const result = await this.modelExecutor(input);
      return result;
    } catch (error) {
      throw new Error(`Model execution failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Gets the name of the model being used.
   * @returns {string} The model name
   */
  public getModelName(): string {
    return this.modelName;
  }

  /**
   * Checks if the model adapter is initialized.
   * @returns {boolean} True if initialized, false otherwise
   */
  public isInitialized(): boolean {
    return this.modelInitialized;
  }

  /**
   * Resets the model adapter state.
   */
  public reset(): void {
    this.modelInitialized = false;
  }
}