/**
 * ModelInterface defines the standard structure and behavior for model classes.
 * This interface ensures consistent implementation of CRUD operations, validation,
 * and error handling across different model implementations.
 * 
 * @template T - The type of the model data
 */
export interface ModelInterface<T> {
    /**
     * Creates a new model instance in the data store.
     * @param data - The data to create the model with
     * @returns Promise resolving to the created model data
     * @throws {Error} If validation fails or creation operation fails
     */
    create(data: Partial<T>): Promise<T>;

    /**
     * Retrieves a model instance by its unique identifier.
     * @param id - The unique identifier of the model
     * @returns Promise resolving to the found model data or null if not found
     * @throws {Error} If the id is invalid or retrieval operation fails
     */
    findById(id: string): Promise<T | null>;

    /**
     * Updates a model instance in the data store.
     * @param id - The unique identifier of the model to update
     * @param updates - The partial data to update the model with
     * @returns Promise resolving to the updated model data
     * @throws {Error} If validation fails, id is invalid, or update operation fails
     */
    update(id: string, updates: Partial<T>): Promise<T>;

    /**
     * Deletes a model instance from the data store.
     * @param id - The unique identifier of the model to delete
     * @returns Promise resolving to true if deletion was successful
     * @throws {Error} If the id is invalid or deletion operation fails
     */
    delete(id: string): Promise<boolean>;

    /**
     * Validates model data against the defined schema/rules.
     * @param data - The data to validate
     * @returns Promise resolving to the validated data
     * @throws {ValidationError} If the data fails validation
     */
    validate(data: Partial<T>): Promise<Partial<T>>;
}

/**
 * Custom error class for validation failures.
 */
export class ValidationError extends Error {
    constructor(message: string, public readonly errors: Record<string, string>) {
        super(message);
        this.name = 'ValidationError';
        Object.setPrototypeOf(this, ValidationError.prototype);
    }
}