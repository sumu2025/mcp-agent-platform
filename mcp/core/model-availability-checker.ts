/**
 * ModelAvailabilityChecker component for checking the availability of models.
 * This class provides methods to verify if a given model is available and ready for use.
 */
class ModelAvailabilityChecker {
  private availableModels: Set<string>;
  private lastCheckedTimestamp: number | null;

  /**
   * Creates an instance of ModelAvailabilityChecker.
   * @param {string[]} initialModels - Optional array of initially available model names.
   */
  constructor(initialModels: string[] = []) {
    this.availableModels = new Set(initialModels);
    this.lastCheckedTimestamp = null;
  }

  /**
   * Checks if a model is available.
   * @param {string} modelName - The name of the model to check.
   * @returns {boolean} True if the model is available, false otherwise.
   * @throws {Error} If modelName is not a string or is empty.
   */
  public isModelAvailable(modelName: string): boolean {
    if (typeof modelName !== 'string' || modelName.trim() === '') {
      throw new Error('Model name must be a non-empty string');
    }

    return this.availableModels.has(modelName);
  }

  /**
   * Adds a model to the available models set.
   * @param {string} modelName - The name of the model to add.
   * @throws {Error} If modelName is not a string or is empty.
   */
  public addAvailableModel(modelName: string): void {
    if (typeof modelName !== 'string' || modelName.trim() === '') {
      throw new Error('Model name must be a non-empty string');
    }

    this.availableModels.add(modelName);
    this.updateLastCheckedTimestamp();
  }

  /**
   * Removes a model from the available models set.
   * @param {string} modelName - The name of the model to remove.
   * @throws {Error} If modelName is not a string or is empty.
   */
  public removeAvailableModel(modelName: string): void {
    if (typeof modelName !== 'string' || modelName.trim() === '') {
      throw new Error('Model name must be a non-empty string');
    }

    this.availableModels.delete(modelName);
    this.updateLastCheckedTimestamp();
  }

  /**
   * Gets all available models.
   * @returns {string[]} An array of available model names.
   */
  public getAvailableModels(): string[] {
    return Array.from(this.availableModels);
  }

  /**
   * Gets the timestamp of the last availability check or update.
   * @returns {number | null} The timestamp in milliseconds or null if never checked.
   */
  public getLastCheckedTimestamp(): number | null {
    return this.lastCheckedTimestamp;
  }

  /**
   * Updates the last checked timestamp to the current time.
   * @private
   */
  private updateLastCheckedTimestamp(): void {
    this.lastCheckedTimestamp = Date.now();
  }

  /**
   * Clears all available models and resets the timestamp.
   */
  public clearAvailableModels(): void {
    this.availableModels.clear();
    this.lastCheckedTimestamp = null;
  }
}

export default ModelAvailabilityChecker;