/**
 * Represents the interface for interacting with AI models.
 * This interface defines the standard methods and properties that
 * an AI model implementation should provide.
 */
interface AIModelInterface {
  /**
   * The unique identifier for the AI model.
   */
  readonly id: string;

  /**
   * The name of the AI model.
   */
  readonly name: string;

  /**
   * The version of the AI model.
   */
  readonly version: string;

  /**
   * Processes input data and returns a prediction or response.
   * @param input - The input data to be processed by the model.
   * @returns A promise that resolves with the model's output.
   * @throws {Error} If the input is invalid or processing fails.
   */
  process(input: unknown): Promise<unknown>;

  /**
   * Validates the input data against the model's requirements.
   * @param input - The input data to validate.
   * @returns A boolean indicating whether the input is valid.
   */
  validateInput(input: unknown): boolean;

  /**
   * Validates the output data to ensure it meets expected standards.
   * @param output - The output data to validate.
   * @returns A boolean indicating whether the output is valid.
   */
  validateOutput(output: unknown): boolean;

  /**
   * Initializes the model with any required setup.
   * @returns A promise that resolves when initialization is complete.
   * @throws {Error} If initialization fails.
   */
  initialize(): Promise<void>;

  /**
   * Cleans up resources used by the model.
   * @returns A promise that resolves when cleanup is complete.
   * @throws {Error} If cleanup fails.
   */
  dispose(): Promise<void>;
}