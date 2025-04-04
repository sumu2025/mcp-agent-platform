/**
 * AIModelFallbackDemo component that demonstrates fallback behavior for AI model operations.
 * This class provides a mechanism to handle primary model failures by falling back to a secondary model.
 */
class AIModelFallbackDemo {
  private primaryModel: AIModel;
  private fallbackModel: AIModel;
  private logger: Logger;

  /**
   * Creates an instance of AIModelFallbackDemo.
   * @param {AIModel} primaryModel - The primary AI model to use for operations.
   * @param {AIModel} fallbackModel - The fallback AI model to use if primary fails.
   * @param {Logger} [logger=console] - Logger instance for error and info logging.
   */
  constructor(
    primaryModel: AIModel,
    fallbackModel: AIModel,
    logger: Logger = console
  ) {
    if (!primaryModel || !fallbackModel) {
      throw new Error('Both primary and fallback models are required');
    }

    this.primaryModel = primaryModel;
    this.fallbackModel = fallbackModel;
    this.logger = logger;
  }

  /**
   * Executes a prediction using the primary model, falling back to the secondary model if needed.
   * @template T - The expected return type of the prediction.
   * @param {ModelInput} input - The input data for the prediction.
   * @returns {Promise<T>} - The prediction result.
   * @throws {Error} - If both primary and fallback models fail.
   */
  async predict<T>(input: ModelInput): Promise<T> {
    try {
      this.validateInput(input);
      const result = await this.primaryModel.predict<T>(input);
      return result;
    } catch (primaryError) {
      this.logger.warn('Primary model failed, attempting fallback', primaryError);

      try {
        const fallbackResult = await this.fallbackModel.predict<T>(input);
        this.logger.info('Successfully used fallback model');
        return fallbackResult;
      } catch (fallbackError) {
        this.logger.error('Both primary and fallback models failed', {
          primaryError,
          fallbackError,
        });
        throw new Error(
          'Prediction failed: both primary and fallback models encountered errors'
        );
      }
    }
  }

  /**
   * Validates the model input data.
   * @private
   * @param {ModelInput} input - The input data to validate.
   * @throws {Error} - If input is invalid.
   */
  private validateInput(input: ModelInput): void {
    if (!input) {
      throw new Error('Input cannot be null or undefined');
    }

    if (typeof input !== 'object' || Array.isArray(input)) {
      throw new Error('Input must be a non-array object');
    }

    if (Object.keys(input).length === 0) {
      throw new Error('Input cannot be empty');
    }
  }
}

/**
 * Interface representing an AI model with prediction capability.
 */
interface AIModel {
  /**
   * Makes a prediction based on the input.
   * @template T - The expected return type of the prediction.
   * @param {ModelInput} input - The input data for the prediction.
   * @returns {Promise<T>} - The prediction result.
   */
  predict<T>(input: ModelInput): Promise<T>;
}

/**
 * Type representing model input data.
 */
type ModelInput = Record<string, unknown>;

/**
 * Interface for logging functionality.
 */
interface Logger {
  info(message?: unknown, ...optionalParams: unknown[]): void;
  warn(message?: unknown, ...optionalParams: unknown[]): void;
  error(message?: unknown, ...optionalParams: unknown[]): void;
}