/**
 * UnifiedModelManager is a centralized manager for handling various models in a unified way.
 * It provides methods for registering, retrieving, and managing models with type safety.
 */
class UnifiedModelManager<T extends { id: string }> {
  private models: Map<string, T>;
  private modelName: string;

  /**
   * Creates an instance of UnifiedModelManager.
   * @param {string} modelName - The name of the model type being managed (for error messages).
   */
  constructor(modelName: string) {
    this.models = new Map();
    this.modelName = modelName;
  }

  /**
   * Registers a new model with the manager.
   * @param {T} model - The model to register.
   * @throws {Error} If the model is invalid or already registered.
   */
  register(model: T): void {
    if (!model || typeof model !== 'object') {
      throw new Error(`Invalid ${this.modelName} model provided`);
    }

    if (!model.id || typeof model.id !== 'string') {
      throw new Error(`${this.modelName} model must have a valid string id`);
    }

    if (this.models.has(model.id)) {
      throw new Error(`${this.modelName} with id '${model.id}' already exists`);
    }

    this.models.set(model.id, model);
  }

  /**
   * Retrieves a model by its ID.
   * @param {string} id - The ID of the model to retrieve.
   * @returns {T | undefined} The model if found, undefined otherwise.
   */
  getById(id: string): T | undefined {
    if (typeof id !== 'string') {
      throw new Error('Model ID must be a string');
    }
    return this.models.get(id);
  }

  /**
   * Retrieves all registered models.
   * @returns {T[]} An array of all registered models.
   */
  getAll(): T[] {
    return Array.from(this.models.values());
  }

  /**
   * Updates an existing model.
   * @param {string} id - The ID of the model to update.
   * @param {Partial<T>} updates - The updates to apply to the model.
   * @throws {Error} If the model doesn't exist or updates are invalid.
   */
  update(id: string, updates: Partial<T>): void {
    if (!this.models.has(id)) {
      throw new Error(`${this.modelName} with id '${id}' not found`);
    }

    if (!updates || typeof updates !== 'object') {
      throw new Error('Invalid updates provided');
    }

    const existingModel = this.models.get(id)!;
    this.models.set(id, { ...existingModel, ...updates });
  }

  /**
   * Removes a model from the manager.
   * @param {string} id - The ID of the model to remove.
   * @returns {boolean} True if the model was removed, false otherwise.
   */
  remove(id: string): boolean {
    return this.models.delete(id);
  }

  /**
   * Clears all models from the manager.
   */
  clear(): void {
    this.models.clear();
  }

  /**
   * Gets the number of registered models.
   * @returns {number} The count of registered models.
   */
  get count(): number {
    return this.models.size;
  }
}

export { UnifiedModelManager };