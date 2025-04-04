/**
 * FallbackDemo component provides a fallback mechanism for handling errors and undefined states.
 * It is designed to be modular, type-safe, and easily testable.
 */
class FallbackDemo<T> {
  private primaryValue: T | undefined;
  private fallbackValue: T;
  private errorHandler: (error: unknown) => void;

  /**
   * Creates an instance of FallbackDemo.
   * @param {T} fallbackValue - The value to use when primary value is not available
   * @param {(error: unknown) => void} [errorHandler=console.error] - Custom error handler function
   */
  constructor(fallbackValue: T, errorHandler: (error: unknown) => void = console.error) {
    this.fallbackValue = fallbackValue;
    this.errorHandler = errorHandler;
  }

  /**
   * Sets the primary value to be used.
   * @param {T} value - The primary value
   * @throws {TypeError} If value is null or undefined
   */
  setPrimaryValue(value: T): void {
    if (value == null) {
      throw new TypeError('Primary value cannot be null or undefined');
    }
    this.primaryValue = value;
  }

  /**
   * Gets the current value, falling back to the fallback value if primary is not available.
   * @returns {T} The current value (primary if available, otherwise fallback)
   */
  getValue(): T {
    return this.primaryValue ?? this.fallbackValue;
  }

  /**
   * Safely executes a function and falls back to default value if execution fails.
   * @param {() => T} fn - Function to execute
   * @returns {T} The result of the function or fallback value if execution fails
   */
  executeWithFallback(fn: () => T): T {
    try {
      const result = fn();
      this.setPrimaryValue(result);
      return result;
    } catch (error) {
      this.errorHandler(error);
      return this.fallbackValue;
    }
  }

  /**
   * Resets the component to its initial state (clears primary value).
   */
  reset(): void {
    this.primaryValue = undefined;
  }

  /**
   * Checks if the component is currently using the fallback value.
   * @returns {boolean} True if using fallback, false otherwise
   */
  isUsingFallback(): boolean {
    return this.primaryValue === undefined;
  }
}

export { FallbackDemo };