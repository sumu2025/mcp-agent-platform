/**
 * UnifiedModelInterface defines a standardized interface for model operations,
 * providing a consistent way to interact with different types of models.
 * 
 * @template T - The type of the model data
 */
export interface UnifiedModelInterface<T> {
    /**
     * Creates a new model instance with the provided data.
     * 
     * @param {Partial<T>} data - The data to create the model with
     * @returns {Promise<T>} A promise that resolves with the created model
     * @throws {Error} If the input data is invalid or creation fails
     */
    create(data: Partial<T>): Promise<T>;

    /**
     * Retrieves a model by its unique identifier.
     * 
     * @param {string} id - The unique identifier of the model
     * @returns {Promise<T | null>} A promise that resolves with the found model or null if not found
     * @throws {Error} If the id is invalid or retrieval fails
     */
    getById(id: string): Promise<T | null>;

    /**
     * Updates a model with the specified identifier.
     * 
     * @param {string} id - The unique identifier of the model to update
     * @param {Partial<T>} updates - The updates to apply to the model
     * @returns {Promise<T | null>} A promise that resolves with the updated model or null if not found
     * @throws {Error} If the id or updates are invalid, or update fails
     */
    update(id: string, updates: Partial<T>): Promise<T | null>;

    /**
     * Deletes a model by its unique identifier.
     * 
     * @param {string} id - The unique identifier of the model to delete
     * @returns {Promise<boolean>} A promise that resolves with true if deletion was successful, false otherwise
     * @throws {Error} If the id is invalid or deletion fails
     */
    delete(id: string): Promise<boolean>;

    /**
     * Lists all models that match the optional filter criteria.
     * 
     * @param {Partial<T>} [filters] - Optional filters to apply to the listing
     * @returns {Promise<T[]>} A promise that resolves with an array of matching models
     * @throws {Error} If filtering fails or data retrieval fails
     */
    list(filters?: Partial<T>): Promise<T[]>;

    /**
     * Validates the model data against the defined schema or rules.
     * 
     * @param {Partial<T>} data - The data to validate
     * @returns {boolean} True if the data is valid, false otherwise
     * @throws {Error} If validation fails due to unexpected errors
     */
    validate(data: Partial<T>): boolean;
}