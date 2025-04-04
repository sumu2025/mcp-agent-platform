/**
 * ModelSwitcher is a component that allows dynamic switching between different models.
 * It provides methods to register, switch, and manage models with type safety and error handling.
 */
class ModelSwitcher<T extends string, M> {
  private models: Map<T, M>;
  private currentModelKey: T | null;

  /**
   * Creates a new ModelSwitcher instance.
   */
  constructor() {
    this.models = new Map();
    this.currentModelKey = null;
  }

  /**
   * Registers a model with a specified key.
   * @param {T} key - The unique identifier for the model.
   * @param {M} model - The model instance to register.
   * @throws {Error} If the key is already registered.
   */
  registerModel(key: T, model: M): void {
    if (this.models.has(key)) {
      throw new Error(`Model with key '${key}' is already registered.`);
    }
    this.models.set(key, model);
  }

  /**
   * Switches to the model identified by the specified key.
   * @param {T} key - The key of the model to switch to.
   * @returns {M} The activated model instance.
   * @throws {Error} If the key is not registered.
   */
  switchModel(key: T): M {
    const model = this.models.get(key);
    if (!model) {
      throw new Error(`Model with key '${key}' is not registered.`);
    }
    this.currentModelKey = key;
    return model;
  }

  /**
   * Gets the currently active model.
   * @returns {M | null} The current model instance, or null if no model is active.
   */
  getCurrentModel(): M | null {
    if (!this.currentModelKey) return null;
    return this.models.get(this.currentModelKey) ?? null;
  }

  /**
   * Gets the key of the currently active model.
   * @returns {T | null} The current model key, or null if no model is active.
   */
  getCurrentModelKey(): T | null {
    return this.currentModelKey;
  }

  /**
   * Checks if a model with the specified key is registered.
   * @param {T} key - The key to check.
   * @returns {boolean} True if the model is registered, false otherwise.
   */
  hasModel(key: T): boolean {
    return this.models.has(key);
  }

  /**
   * Removes a model from the switcher.
   * @param {T} key - The key of the model to remove.
   * @throws {Error} If trying to remove the currently active model.
   */
  removeModel(key: T): void {
    if (this.currentModelKey === key) {
      throw new Error(`Cannot remove currently active model '${key}'. Switch to another model first.`);
    }
    this.models.delete(key);
  }

  /**
   * Clears all registered models and resets the current model.
   */
  clearModels(): void {
    this.models.clear();
    this.currentModelKey = null;
  }

  /**
   * Gets all registered model keys.
   * @returns {T[]} An array of all registered model keys.
   */
  getRegisteredModelKeys(): T[] {
    return Array.from(this.models.keys());
  }
}

export { ModelSwitcher };