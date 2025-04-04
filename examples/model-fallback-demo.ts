/**
 * ModelFallbackDemo component that provides fallback functionality for model operations.
 * This class handles scenarios where primary model operations fail by providing
 * alternative fallback mechanisms with proper error handling and type safety.
 */
class ModelFallbackDemo<T> {
  private primaryModel: T | null;
  private fallbackModel: T | null;
  private isPrimaryActive: boolean;

  /**
   * Creates an instance of ModelFallbackDemo.
   * @param {T} primaryModel - The primary model instance to use
   * @param {T} fallbackModel - The fallback model instance to use when primary fails
   */
  constructor(primaryModel: T, fallbackModel: T) {
    if (!primaryModel || !fallbackModel) {
      throw new Error('Both primary and fallback models must be provided');
    }

    this.primaryModel = primaryModel;
    this.fallbackModel = fallbackModel;
    this.isPrimaryActive = true;
  }

  /**
   * Executes an operation using the active model (primary or fallback)
   * @template R - The return type of the operation
   * @param {(model: T) => R} operation - The operation to perform on the model
   * @returns {R} The result of the operation
   * @throws {Error} When both primary and fallback operations fail
   */
  execute<R>(operation: (model: T) => R): R {
    try {
      const activeModel = this.isPrimaryActive ? this.primaryModel : this.fallbackModel;
      if (!activeModel) {
        throw new Error('No active model available');
      }

      return operation(activeModel);
    } catch (primaryError) {
      if (this.isPrimaryActive) {
        console.warn('Primary model failed, falling back to secondary', primaryError);
        this.isPrimaryActive = false;
        return this.execute(operation);
      }

      throw new Error(
        `Both primary and fallback model operations failed. Last error: ${primaryError instanceof Error ? primaryError.message : String(primaryError)}`
      );
    }
  }

  /**
   * Resets the fallback state to use the primary model again
   */
  resetToPrimary(): void {
    this.isPrimaryActive = true;
  }

  /**
   * Gets the currently active model
   * @returns {T | null} The active model instance
   */
  getActiveModel(): T | null {
    return this.isPrimaryActive ? this.primaryModel : this.fallbackModel;
  }

  /**
   * Checks if the primary model is currently active
   * @returns {boolean} True if primary model is active
   */
  isUsingPrimary(): boolean {
    return this.isPrimaryActive;
  }
}

export { ModelFallbackDemo };