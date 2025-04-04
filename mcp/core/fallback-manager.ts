/**
 * FallbackManager is a utility class that manages fallback strategies for operations
 * that may fail. It provides a way to attempt primary operations and gracefully
 * fall back to secondary operations when the primary ones fail.
 */
export class FallbackManager<T> {
  private primaryAction: () => Promise<T>;
  private fallbackActions: Array<() => Promise<T>>;
  private shouldRetryPrimary: boolean;
  private maxRetries: number;

  /**
   * Creates an instance of FallbackManager.
   * @param {() => Promise<T>} primaryAction - The primary action to attempt first.
   * @param {Array<() => Promise<T>>} fallbackActions - Array of fallback actions to try if primary fails.
   * @param {boolean} [shouldRetryPrimary=false] - Whether to retry the primary action before falling back.
   * @param {number} [maxRetries=1] - Maximum number of retries for the primary action (if shouldRetryPrimary is true).
   * @throws {Error} If primaryAction or fallbackActions are not provided or invalid.
   */
  constructor(
    primaryAction: () => Promise<T>,
    fallbackActions: Array<() => Promise<T>> = [],
    shouldRetryPrimary: boolean = false,
    maxRetries: number = 1
  ) {
    if (typeof primaryAction !== 'function') {
      throw new Error('Primary action must be a function');
    }

    if (!Array.isArray(fallbackActions) || fallbackActions.some(action => typeof action !== 'function')) {
      throw new Error('Fallback actions must be an array of functions');
    }

    if (maxRetries < 0) {
      throw new Error('Max retries must be a non-negative number');
    }

    this.primaryAction = primaryAction;
    this.fallbackActions = fallbackActions;
    this.shouldRetryPrimary = shouldRetryPrimary;
    this.maxRetries = maxRetries;
  }

  /**
   * Executes the fallback strategy by attempting the primary action first,
   * then falling back to secondary actions if needed.
   * @returns {Promise<T>} The result of either the primary or fallback action.
   * @throws {Error} If all actions fail.
   */
  public async execute(): Promise<T> {
    let lastError: Error | null = null;

    // Try primary action with potential retries
    if (this.shouldRetryPrimary) {
      for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
        try {
          return await this.primaryAction();
        } catch (error) {
          lastError = error instanceof Error ? error : new Error(String(error));
          if (attempt === this.maxRetries) {
            break; // Stop retrying after max attempts
          }
        }
      }
    } else {
      // Try primary action once
      try {
        return await this.primaryAction();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
      }
    }

    // Try fallback actions in order
    for (const fallbackAction of this.fallbackActions) {
      try {
        return await fallbackAction();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        continue; // Try next fallback
      }
    }

    // All attempts failed
    throw new Error('All fallback strategies failed', { cause: lastError });
  }

  /**
   * Adds a new fallback action to the end of the fallback sequence.
   * @param {() => Promise<T>} action - The fallback action to add.
   * @throws {Error} If the action is not a function.
   */
  public addFallbackAction(action: () => Promise<T>): void {
    if (typeof action !== 'function') {
      throw new Error('Fallback action must be a function');
    }
    this.fallbackActions.push(action);
  }

  /**
   * Clears all fallback actions.
   */
  public clearFallbackActions(): void {
    this.fallbackActions = [];
  }

  /**
   * Updates the primary action.
   * @param {() => Promise<T>} action - The new primary action.
   * @throws {Error} If the action is not a function.
   */
  public setPrimaryAction(action: () => Promise<T>): void {
    if (typeof action !== 'function') {
      throw new Error('Primary action must be a function');
    }
    this.primaryAction = action;
  }

  /**
   * Updates the retry configuration.
   * @param {boolean} shouldRetry - Whether to retry the primary action.
   * @param {number} [maxRetries=1] - Maximum number of retries.
   * @throws {Error} If maxRetries is negative.
   */
  public setRetryConfig(shouldRetry: boolean, maxRetries: number = 1): void {
    if (maxRetries < 0) {
      throw new Error('Max retries must be a non-negative number');
    }
    this.shouldRetryPrimary = shouldRetry;
    this.maxRetries = maxRetries;
  }
}