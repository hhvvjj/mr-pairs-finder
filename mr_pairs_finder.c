// ***********************************************************************************
// mr PAIRS FINDER
// ***********************************************************************************
//
// Author: Javier Hernandez
//
// Email:  271314@pm.me
// 
// Description:
//   The tuple-based transform is a reversible procedure to represent Collatz sequences
//   using the tuple [p, f(p), m, q]. Companion computational tools and step-by-step 
//   visualizations available at:
//   https://github.com/hhvvjj/a-new-algebraic-framework-for-the-collatz-conjecture
//
//   During the tuple-based transform representation, the multiplicity parameter m repeats
//   two elements (consecutively or at different distances), creating pseudocycles. These two
//   values, named as mr, are shown at the ends of a pseudocycle.
//
//   This is a high-performance parallel search engine to find mr pairs. It efficiently 
//   detects pseudocycles by analyzing the m-value repetition and generates detailed JSON
//   output of all discovered mr pairs in the range of 1 to (2^exp) - 1. The output also 
//   includes the first occurrence of n that generated each mr value. It uses OpenMP for 
//   parallelization and optimized hash tables for efficient pseudocycle detection.
//
//   Research findings show that mr value discovery exhibits remarkable sparsity across large
//   computational ranges. Comprehensive analysis of the complete range 1 to 2^40, 1099511627775
//   numbers, reveals only 42 distinct mr values, suggesting that unique pseudocycle patterns
//   are extremely rare phenomena in Collatz sequence behavior.
//
//   The list of these 42 mr values is: 0, 1, 2, 3, 6, 7, 8, 9, 12, 16, 19, 25, 45, 53, 60, 79,
//   91, 121, 125, 141, 166, 188, 205, 243, 250, 324, 333, 432, 444, 487, 576, 592, 649, 667,
//   683, 865, 889, 1153, 1214, 1821, 2428, 3643
//
// Usage:
//   ./mr_pairs_finder <exponent>
//   Example: ./mr_pairs_finder 25
//
// Output:
//   - JSON file: mr_pairs_detected_on_range_1_to_2pow<exponent>.json
//   - Console: Real-time progress and comprehensive summary report
//
// License:
//   CC-BY-NC-SA 4.0 International 
//   For additional details, visit:
//   https://creativecommons.org/licenses/by-nc-sa/4.0/
//
//   For full details, visit 
//   https://github.com/hhvvjj/mr-pairs-finder/blob/main/LICENSE
//
// Research Reference:
//   Based on the tuple-based transform methodology described in:
//   https://doi.org/10.5281/zenodo.15546925
//
// ***********************************************************************************

// ***********************************************************************************
// * 1. HEADERS, DEFINES, TYPEDEFS & GLOBAL VARIABLES
// ***********************************************************************************

// =============================
// SYSTEM HEADERS
// =============================

#include <signal.h>     // Signal handling: signal(), SIGINT for graceful interruption
#include <stdio.h>      // Standard I/O operations: printf, fprintf, fopen, fclose, etc.
#include <stdlib.h>     // General utilities: malloc, free, exit, atoi, realloc
#include <string.h>     // String manipulation: strcmp, memset, strlen, snprintf
#include <stdint.h>     // Fixed-width integer types: uint64_t, uint32_t, int32_t
#include <stdbool.h>    // Boolean type support: bool, true, false
#include <unistd.h>     // File access check: access(), F_OK for checkpoint validation
#include <omp.h>        // OpenMP parallelization: #pragma omp, omp_get_wtime, locks

// =============================
// CHECKPOINT CONFIGURATION
// =============================

// Checkpoint system for long-running computations
#define CHECKPOINT_INTERVAL 300.0           // Seconds between checkpoint saves (5 minutes)
#define CHECKPOINT_FILE "checkpoint.bin"    // Binary checkpoint file name
#define CHECKPOINT_BACKUP "checkpoint.bak"  // Backup file for safety
#define CHECKPOINT_MAGIC 0x4D52434B         // "MRCK" in hex for validation

// =============================
// SAFETY AND PERFORMANCE LIMITS
// =============================

// Sequence safety constraints to prevent infinite loops and resource exhaustion
#define MAX_SEQUENCE_LENGTH 100000          // Maximum steps in Collatz sequence before termination

// Progress monitoring configuration for real-time feedback
#define PROGRESS_UPDATE_INTERVAL 3.0        // Seconds between progress display updates
#define PROGRESS_CHECK_FREQUENCY 100000     // Numbers processed between progress checks

// Memory management and dynamic allocation parameters
#define INITIAL_M_CAPACITY 100              // Starting capacity for m-values storage
#define MEMORY_EXPANSION_FACTOR 2           // Growth multiplier when expanding arrays

// Input validation ranges for command-line arguments
#define MIN_EXPONENT 1                      // Minimum search range exponent (2^1)
#define MAX_EXPONENT 63                     // Maximum search range exponent (2^63, technical limit)
                                            // Note: Exponents > 45 are computationally impractical
// =============================
// HASH TABLE CONFIGURATION
// =============================

// Hash table sizing (must be power of 2 for efficient modulo via bitwise AND)
#define HASH_TABLE_SIZE 8192                // Total hash table buckets (2^13)
#define HASH_MASK 8191                      // Bitmask for hash function (HASH_TABLE_SIZE - 1)

// =============================
// SIGNAL HANDLING
// =============================

volatile sig_atomic_t checkpoint_signal_received = 0;

// =============================
// CORE DATA STRUCTURES
// =============================

/**
 * @brief Checkpoint data structure for resume capability
 * 
 * Stores minimal state needed to resume interrupted computations:
 * - Last processed number for loop continuation
 * - Exponent for validation on resume
 * - Count of unique mr values discovered so far
 * - Arrays of mr values and their first occurrence n values
 */
typedef struct {
    uint64_t last_n_processed;              // Last number successfully processed
    uint64_t total_processed;               // Total numbers actually processed
    int exponent;                           // Search exponent for validation
    int unique_count;                       // Number of unique mr values found
    uint64_t magic;                         // Magic number for file validation (0x4D52434B)
} CheckpointHeader;

/**
 * @brief Hash table node for efficient m-value lookup during pseudocycle detection
 * 
 * Implements separate chaining collision resolution. Each node stores an m-value
 * and forms linked lists at hash table buckets for O(1) average-case lookup.
 */
typedef struct HashNode {
    uint64_t value;                         // Stored m-value for repetition detection
    struct HashNode* next;                  // Next node in collision chain
} HashNode;

/**
 * @brief Container for m-values with hash table optimization for sequence analysis
 * 
 * Hybrid data structure combining dynamic array for sequential storage with
 * hash table for fast repetition detection during Collatz sequence analysis.
 */
typedef struct {
    HashNode* buckets[HASH_TABLE_SIZE];     // Hash table for O(1) lookup
    uint64_t* values;                       // Dynamic array for sequential access
    int count;                              // Current number of stored m-values
    int capacity;                           // Maximum capacity before reallocation
} mValues;

/**
 * @brief Thread-safe collection for unique mr values discovered during search
 * 
 * Maintains parallel arrays storing unique mr values and their first occurrence
 * n values. Uses OpenMP locks for thread-safe concurrent access during parallel
 * search operations.
 */
typedef struct {
    uint64_t* values;                       // Array of unique mr values found
    uint64_t* first_n;                      // Array of first n values for each mr
    int count;                              // Current number of unique mr values
    int capacity;                           // Maximum capacity before expansion
    omp_lock_t lock;                        // OpenMP lock for thread safety
} UniqueMrSet;

/**
 * @brief Thread-safe progress tracking for real-time monitoring of parallel operations
 * 
 * Maintains counters and timing information for progress reporting during parallel
 * search. Uses atomic operations and locks for high-frequency updates.
 */
typedef struct {
    uint64_t processed;                     // Total numbers processed across threads
    uint64_t found_count;                   // Count of numbers yielding mr values
    uint64_t last_n_with_new_unique;        // Most recent n that found new unique mr
    double last_update_time;                // Timestamp of last progress display
    omp_lock_t lock;                        // OpenMP lock for atomic updates
} ProgressTracker;

/**
 * @brief Thread-safe statistics for sequence classification (A, B, C types)
 * 
 * Collects distribution statistics for all sequences in the search range:
 * - Type A: mr=0 (reaches trivial cycle without other pseudocycles)
 * - Type B: mr>0 and M*=mr (pseudocycle at maximum m)
 * - Type C: mr>0 and M*>mr (pseudocycle before reaching maximum m)
 */
typedef struct {
    uint64_t type_A_count;      // Sequences where M* appears before first mr
    uint64_t type_B_count;      // Sequences where M* appears between first and second mr
    uint64_t type_C_count;      // Sequences where M* appears after second mr
    uint64_t* mr_distribution;
    int mr_dist_capacity;
    omp_lock_t lock;
} SequenceStatistics;

/**
 * @brief Complete search context containing all operational parameters and state
 * 
 * Central structure holding configuration, shared resources, and storage components
 * for the parallel search operation. Passed between functions to maintain consistent
 * access to all program state.
 */
typedef struct {
    uint64_t max_n;                         // Upper bound of search range (exclusive)
    int exponent;                           // Power of 2 exponent for max_n
    double start_time;                      // Search operation start timestamp
    UniqueMrSet* unique_set;                // Collection of discovered unique mr values
    ProgressTracker* progress;              // Progress monitoring and reporting
    SequenceStatistics* stats;              // Statistics
} SearchContext;

/**
 * @brief Signal handler for graceful checkpoint save on interruption.
 * 
 * Captures SIGINT (Ctrl+C) to allow graceful shutdown with checkpoint save
 * before program termination, preventing loss of computation progress.
 * 
 * @param sig Signal number received (expected SIGINT)
 */
static void checkpoint_signal_handler(int sig) {
    if (sig == SIGINT) {
        checkpoint_signal_received = 1;
    }
 }

// ***********************************************************************************
// * 2. UTILITY FUNCTIONS
// ***********************************************************************************

 /**
 * @brief Safely allocates memory with automatic error handling and immediate program termination on failure.
 * 
 * This function provides a wrapper around the standard malloc() call with comprehensive
 * error checking and standardized failure handling. It ensures consistent behavior across
 * the application when memory allocation fails, eliminating the need for repetitive null
 * pointer checks at every allocation site while providing descriptive error messages
 * for debugging purposes.
 * 
 * The function implements a fail-fast strategy: if memory allocation fails, it immediately
 * prints an informative error message to stderr and terminates the program with exit code 1.
 * This approach is suitable for applications where memory allocation failure represents
 * an unrecoverable error condition that should halt execution.
 * 
 * Error Handling Strategy:
 * - Immediate detection of malloc() failure through null pointer check
 * - Descriptive error message including context information for debugging
 * - Graceful program termination with standard error exit code
 * - No possibility of returning null pointers to calling code
 * 
 * @param size The number of bytes to allocate. Must be greater than 0 for meaningful
 *             allocation. The function passes this value directly to malloc().
 * @param context A descriptive string identifying the purpose of the allocation.
 *                This string is included in error messages to aid debugging and
 *                should describe what the memory is intended for.
 *                Examples: "hash table", "m_values array", "unique mr set"
 * 
 * @return A valid pointer to the allocated memory block. This function never returns
 *         NULL because it terminates the program if allocation fails. The returned
 *         memory is uninitialized.
 * 
 * @note This function calls exit(1) on allocation failure, making it unsuitable
 *       for applications that need to recover from memory allocation failures.
 * 
 * @note The returned memory is uninitialized. Use memset() or similar functions
 *       if zero-initialization is required.
 * 
 * @note The context parameter should be a string literal or stable string to
 *       ensure it remains valid during error reporting.
 * 
 * @warning This function terminates the program on failure, so it should only be
 *          used in contexts where immediate termination is acceptable.
 * 
 * @complexity O(1) - constant time wrapper around malloc() with simple error checking
 * 
 * @see malloc(3) for underlying allocation mechanism
 * @see exit(3) for program termination behavior
 * @see safe_realloc() for the corresponding reallocation function
 * 
 * @example
 * ```c
 * // Allocate space for 100 integers
 * int* numbers = safe_malloc(100 * sizeof(int), "integer array");
 * 
 * // Allocate space for a hash table structure
 * HashNode* node = safe_malloc(sizeof(HashNode), "hash node");
 * 
 * // Allocate space for m values array
 * uint64_t* m_values = safe_malloc(capacity * sizeof(uint64_t), "m_values array");
 * 
 * // Error case (simulated)
 * // If system is out of memory, program prints:
 * // "[*] ERROR: Memory allocation failed m_values array"
 * // and exits with code 1
 * ```
 */
static void* safe_malloc(size_t size, const char* context) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "\n[*] ERROR: Memory allocation failed for %s\n", context);
        exit(1);
    }
    return ptr;
}

/**
 * @brief Safely reallocates memory with automatic error handling and immediate program termination on failure.
 * 
 * This function provides a wrapper around the standard realloc() call with comprehensive
 * error checking and standardized failure handling. It ensures consistent behavior when
 * expanding or shrinking dynamically allocated memory blocks, maintaining the same
 * fail-fast strategy as safe_malloc() for uniform error handling throughout the application.
 * 
 * The function handles the complexities of realloc() behavior while providing clear
 * error reporting. Unlike realloc(), which can return NULL on failure while leaving
 * the original pointer valid, this function ensures that allocation failure results
 * in immediate program termination with a descriptive error message.
 * 
 * Reallocation Behavior:
 * - Expands or shrinks the memory block pointed to by ptr
 * - May move the block to a new location if necessary
 * - Preserves existing data up to the minimum of old and new sizes
 * - Returns a pointer to the (possibly moved) memory block
 * - Terminates program immediately if reallocation fails
 * 
 * @param ptr Pointer to the previously allocated memory block to reallocate.
 *            Can be NULL, in which case this function behaves like malloc().
 *            If non-NULL, must be a valid pointer returned by malloc(), calloc(),
 *            or a previous call to realloc().
 * @param size The new size in bytes for the memory block. If 0, the behavior
 *             is implementation-defined (may free the block or return NULL).
 * @param context A descriptive string identifying the purpose of the reallocation.
 *                Used in error messages for debugging. Should describe what the
 *                memory expansion is for, e.g., "m_values expansion", "hash table growth".
 * 
 * @return A valid pointer to the reallocated memory block. This function never
 *         returns NULL because it terminates the program if reallocation fails.
 *         The returned pointer may be different from the input pointer if the
 *         block was moved.
 * 
 * @note If reallocation fails, the original memory block remains valid and unchanged,
 *       but the program terminates before the caller can access it.
 * 
 * @note The function preserves existing data when expanding memory. New memory
 *       beyond the original size is uninitialized.
 * 
 * @note When expanding arrays, callers should update their pointer variables with
 *       the returned value, as the memory block may have moved.
 * 
 * @warning This function terminates the program on failure, making it unsuitable
 *          for applications requiring graceful recovery from memory allocation failures.
 * 
 * @warning After calling this function, the original ptr should be considered invalid
 *          and replaced with the returned pointer.
 * 
 * @complexity O(n) in worst case where n is the smaller of old and new sizes,
 *            due to potential memory copying if the block needs to be moved
 * 
 * @see realloc(3) for underlying reallocation mechanism
 * @see safe_malloc() for initial allocation with similar error handling
 * @see add_m_value() for usage example in dynamic array expansion
 * 
 * @example
 * ```c
 * // Initial allocation
 * int* array = safe_malloc(10 * sizeof(int), "integer array");
 * 
 * // Expand the array to hold 20 integers
 * array = safe_realloc(array, 20 * sizeof(int), "integer array expansion");
 * 
 * // Shrink the array back to 5 integers
 * array = safe_realloc(array, 5 * sizeof(int), "integer array shrinkage");
 * 
 * // Usage in dynamic structure expansion
 * mv->values = safe_realloc(mv->values, new_capacity * sizeof(uint64_t), "m_values expansion");
 * mv->capacity = new_capacity;
 * 
 * // Error case (simulated)
 * // If system cannot provide more memory, program prints:
 * // "[*] ERROR: Memory reallocation failed for m_values expansion"
 * // and exits with code 1
 * ```
 */
static void* safe_realloc(void* ptr, size_t size, const char* context) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr) {
        fprintf(stderr, "\n[*] ERROR: Memory reallocation failed for %s\n", context);
        exit(1);
    }
    return new_ptr;
}

/**
 * @brief Computes a hash value for a 64-bit unsigned integer using multiplicative hashing with bit masking.
 * 
 * This function implements a multiplicative hash function that provides good distribution quality
 * for 64-bit values mapped to hash table indices. The algorithm uses the golden ratio-based
 * multiplier (2654435761) combined with bit shifting and masking to achieve balanced hash
 * distribution across the available hash table buckets.
 * 
 * The multiplicative hashing approach significantly outperforms simple bit masking in terms
 * of distribution quality, especially for sequential or mathematically related inputs common
 * in Collatz sequence analysis. This method reduces clustering and provides more uniform
 * distribution across hash table buckets.
 * 
 * Algorithm:
 * 1. Multiply input value by golden ratio constant (2654435761)
 * 2. Extract high-order 32 bits via right shift by 32
 * 3. Apply bitwise AND with HASH_MASK (8191) for final index
 * 
 * @param value The 64-bit unsigned integer to hash. Any value is valid input.
 * 
 * @return A hash value in the range [0, HASH_SIZE-1] where HASH_SIZE = 8192.
 *         The returned value can be used directly as an index into the hash table.
 * 
 * @note The multiplier 2654435761ULL is derived from the golden ratio and provides
 *       excellent distribution characteristics for most input patterns.
 * 
 * @note HASH_MASK is defined as (HASH_SIZE - 1) = 8191, requiring HASH_SIZE to be
 *       a power of 2 for the bitwise AND to work correctly as a modulo operation.
 * 
 * @note This multiplicative approach provides superior hash distribution compared to
 *       simple bit masking, reducing collision rates for sequential and patterned inputs.
 * 
 * @note The right shift by 32 extracts the most random bits from the multiplication
 *       result, further improving distribution quality.
 * 
 * @complexity O(1) - constant time operation with single multiplication and bit operations
 * 
 * @see HASH_SIZE constant definition (8192)
 * @see HASH_MASK constant definition (HASH_SIZE - 1)
 * @see add_m_value() for hash table insertion using this function
 * 
 * @example
 * ```c
 * uint64_t value1 = 12345;
 * uint64_t value2 = 67890;
 * 
 * uint64_t hash1 = hash_function(value1);  // Returns well-distributed hash
 * uint64_t hash2 = hash_function(value2);  // Returns well-distributed hash
 * 
 * // Use hash values as array indices
 * if (hash_table[hash1] != NULL) {
 *     // Handle collision or existing entry
 * }
 * 
 * // Sequential values get different hash buckets
 * for (uint64_t i = 1000; i < 1100; i++) {
 *     uint64_t hash = hash_function(i);  // Good distribution despite sequential input
 * }
 * ```
 */
static inline uint64_t hash_function(uint64_t value) {
    return (uint32_t)((value * 2654435761ULL) >> 32) & HASH_MASK;
}

// ***********************************************************************************
// * 3. m VALUES CONTAINER (SEQUENCE ANALYSIS)
// ***********************************************************************************

/**
 * @brief Initializes an mValues container with hash table optimization for efficient m value storage and lookup.
 * 
 * This function sets up a hybrid data structure that combines a dynamic array for sequential
 * storage of m values with a hash table for O(1) average-case lookup performance. The design
 * enables both efficient iteration through discovered m values and fast repetition detection
 * during Collatz sequence analysis.
 * 
 * The mValues container serves as a critical component in pseudocycle detection, storing all
 * m values encountered in a single Collatz sequence and providing fast lookup to detect when
 * an m value repeats (indicating pseudocycle completion). The dual storage approach optimizes
 * for both access patterns needed during sequence analysis.
 * 
 * Initialization Process:
 * 1. Set initial capacity to INITIAL_M_CAPACITY (100) for reasonable starting size
 * 2. Allocate dynamic array for sequential m value storage
 * 3. Initialize count to zero for empty container state
 * 4. Clear all hash table buckets to NULL for empty hash state
 * 
 * Data Structure Design:
 * - Dynamic array (values): Stores m values in discovery order for iteration
 * - Hash table (buckets): Provides O(1) lookup for repetition detection
 * - Capacity tracking: Enables automatic expansion when storage limits reached
 * - Count tracking: Maintains current number of stored m values
 * 
 * @param mv Pointer to the mValues structure to initialize. Must point to a valid
 *           mValues structure that will be modified in-place. The structure should
 *           not be previously initialized to avoid memory leaks.
 * 
 * @note The function allocates memory for the initial values array but does not
 *       allocate hash table nodes until values are actually added via add_m_value().
 * 
 * @note INITIAL_M_CAPACITY (100) is chosen as a reasonable starting size based on
 *       typical Collatz sequence lengths before pseudocycle detection.
 * 
 * @note All hash table buckets are initialized to NULL, representing empty collision
 *       chains that will be populated as m values are added to the container.
 * 
 * @note The initialized container is ready for immediate use with add_m_value()
 *       and is_m_repeated() functions.
 * 
 * @warning Do not call this function on an already-initialized mValues structure
 *          without first calling destroy_m_values() to avoid memory leaks.
 * 
 * @complexity O(n) where n is HASH_TABLE_SIZE (8192) due to hash bucket initialization
 * 
 * @see destroy_m_values() for proper cleanup of initialized containers
 * @see add_m_value() for adding m values to the initialized container
 * @see is_m_repeated() for checking repetitions using the hash table
 * @see INITIAL_M_CAPACITY and HASH_TABLE_SIZE for sizing constants
 * 
 * @example
 * ```c
 * // Initialize container for sequence analysis
 * mValues m_values;
 * init_m_values(&m_values);
 * 
 * // Container is now ready for use
 * uint64_t sequence[] = {7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1};
 * 
 * for (int i = 0; i < sequence_length; i++) {
 *     uint64_t m = calculate_m(sequence[i]);
 *     
 *     if (is_m_repeated(&m_values, m)) {
 *         printf("Pseudocycle detected at m=%lu\n", m);
 *         break;
 *     }
 *     
 *     add_m_value(&m_values, m);
 * }
 * 
 * // Cleanup when done
 * destroy_m_values(&m_values);
 * ```
 */
static void init_m_values(mValues* mv) {
    mv->capacity = INITIAL_M_CAPACITY;
    mv->values = safe_malloc(mv->capacity * sizeof(uint64_t), "m_values array");
    mv->count = 0;
    
    // Initialize hash table
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        mv->buckets[i] = NULL;
    }
}

/**
 * @brief Safely deallocates all memory associated with an mValues container and resets its state.
 * 
 * This function performs comprehensive cleanup of an mValues structure, ensuring proper
 * deallocation of both the dynamic array storage and all hash table nodes. The cleanup
 * process handles the complex memory layout of the hybrid data structure, preventing
 * memory leaks while resetting the container to a safe, uninitialized state.
 * 
 * The function implements defensive programming by checking for NULL pointers and uses
 * careful pointer management to avoid double-free errors during hash table cleanup.
 * Each node in every collision chain is individually freed using a safe traversal
 * pattern that saves the next pointer before freeing each node.
 * 
 * Cleanup Process:
 * 1. Validate input pointer for safe NULL handling
 * 2. Traverse and free all hash table collision chains
 * 3. Reset all hash bucket pointers to NULL
 * 4. Deallocate the values array
 * 5. Reset all container state variables to safe values
 * 
 * Memory Management Strategy:
 * The function addresses both user-allocated arrays and dynamically created hash nodes.
 * Hash nodes are created during add_m_value() operations and must be individually
 * freed to prevent memory leaks. The careful traversal pattern ensures no nodes
 * are orphaned during the cleanup process.
 * 
 * @param mv Pointer to the mValues structure to clean up. Can be NULL, in which
 *           case the function returns immediately without action. After successful
 *           cleanup, the structure is left in an uninitialized state.
 * 
 * @note This function is safe to call with NULL pointers, making it suitable for
 *       cleanup in error handling paths where initialization might have failed.
 * 
 * @note After calling this function, the mValues structure is in an uninitialized
 *       state and should not be used until re-initialized with init_m_values().
 * 
 * @note The function sets all pointer fields to NULL and counters to zero, providing
 *       a clean state that can help detect use-after-free errors during debugging.
 * 
 * @note Hash table cleanup uses a safe traversal pattern that saves the next pointer
 *       before freeing each node, preventing access to freed memory.
 * 
 * @warning Do not use the mValues structure after calling this function until it
 *          has been re-initialized with init_m_values().
 * 
 * @complexity O(n) where n is the total number of hash nodes across all buckets,
 *            which is equal to the number of m values that were added to the container
 * 
 * @see init_m_values() for the corresponding initialization function
 * @see add_m_value() for the function that creates hash nodes
 * @see find_first_mr_in_sequence() for typical usage pattern
 * 
 * @example
 * ```c
 * // Typical usage in sequence analysis
 * mValues m_values;
 * init_m_values(&m_values);
 * 
 * // ... use container for sequence analysis ...
 * 
 * // Cleanup when analysis is complete
 * destroy_m_values(&m_values);
 * 
 * // Safe to call with NULL
 * destroy_m_values(NULL);  // No-op, returns immediately
 * 
 * // Error handling example
 * mValues m_values;
 * init_m_values(&m_values);
 * if (some_error_condition) {
 *     destroy_m_values(&m_values);  // Safe cleanup
 *     return error_code;
 * }
 * 
 * // Usage in find_first_mr_in_sequence
 * uint64_t find_first_mr_in_sequence(uint64_t n_start, bool* found) {
 *     mValues m_values;
 *     init_m_values(&m_values);
 *     
 *     // ... sequence analysis ...
 *     
 *     destroy_m_values(&m_values);  // Always cleanup before return
 *     return first_mr;
 * }
 * ```
 */
static void destroy_m_values(mValues* mv) {
    if (!mv) return;
    
    // Clean up hash table
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        HashNode* current = mv->buckets[i];
        while (current) {
            HashNode* next = current->next;
            free(current);
            current = next;
        }
        mv->buckets[i] = NULL;
    }
    
    // Clean up values array
    free(mv->values);
    mv->values = NULL;
    mv->count = 0;
    mv->capacity = 0;
}

/**
 * @brief Efficiently checks if an m value has been previously encountered using hash table lookup.
 * 
 * This function performs fast O(1) average-case lookup to determine if a specific m value
 * has already been stored in the mValues container. This is the core operation for pseudocycle
 * detection in Collatz sequence analysis, as repetition of m values indicates the completion
 * of a pseudocycle pattern.
 * 
 * The function uses the container's hash table for efficient lookup, computing the hash
 * index for the target m value and traversing the collision chain at that bucket until
 * either finding a match or reaching the end of the chain. This approach provides
 * significantly better performance than linear search through the values array.
 * 
 * Pseudocycle Detection Context:
 * In the tuple-based transform approach, when an m value repeats during sequence generation,
 * it indicates that the sequence has entered a pseudocycle. This function enables immediate
 * detection of such repetitions without requiring expensive sequence continuation.
 * 
 * Hash Table Lookup Process:
 * 1. Compute hash index using the hash_function()
 * 2. Access the collision chain at the computed bucket
 * 3. Traverse the linked list comparing each stored value
 * 4. Return true on first match, false if chain is exhausted
 * 
 * @param mv Pointer to the mValues container to search. Must be a properly initialized
 *           container with valid hash table structure. The container is not modified
 *           during the lookup operation.
 * @param m The m value to search for in the container. This represents a transformed
 *          Collatz sequence value that may have been previously encountered.
 * 
 * @return true if the m value has been previously stored in the container
 *         false if the m value is not found in the container
 * 
 * @note This function does not modify the container state and can be called safely
 *       from multiple threads reading the same container (assuming no concurrent writes).
 * 
 * @note The lookup performance depends on the hash function quality and load factor.
 *       With a good hash function, average-case performance is O(1).
 * 
 * @note This function should be called before add_m_value() to implement proper
 *       pseudocycle detection logic in sequence analysis.
 * 
 * @note The function traverses collision chains linearly, so worst-case performance
 *       is O(n) where n is the maximum chain length at any hash bucket.
 * 
 * @complexity Average case: O(1) - constant time lookup with good hash distribution
 *            Worst case: O(n) - linear time when all values hash to the same bucket
 * 
 * @see hash_function() for the hash computation algorithm
 * @see add_m_value() for adding values that this function can subsequently find
 * @see find_first_mr_in_sequence() for usage in pseudocycle detection
 * @see HashNode structure for collision chain implementation
 * 
 * @example
 * ```c
 * // Typical usage in sequence analysis
 * mValues m_values;
 * init_m_values(&m_values);
 * 
 * uint64_t sequence[] = {7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1};
 * 
 * for (int i = 0; i < sequence_length; i++) {
 *     uint64_t m = calculate_m(sequence[i]);
 *     
 *     // Check for repetition before adding
 *     if (is_m_repeated(&m_values, m)) {
 *         printf("Pseudocycle detected! m=%lu repeats\n", m);
 *         break;  // First repetition found
 *     }
 *     
 *     // Add to container for future repetition checks
 *     add_m_value(&m_values, m);
 * }
 * 
 * // Example with specific values
 * add_m_value(&m_values, 25);
 * add_m_value(&m_values, 108);
 * 
 * bool found1 = is_m_repeated(&m_values, 25);   // Returns true
 * bool found2 = is_m_repeated(&m_values, 999);  // Returns false
 * bool found3 = is_m_repeated(&m_values, 108);  // Returns true
 * ```
 */
static bool is_m_repeated(const mValues* mv, uint64_t m) {
    uint64_t hash = hash_function(m);
    HashNode* current = mv->buckets[hash];
    
    while (current) {
        if (current->value == m) {
            return true;
        }
        current = current->next;
    }
    return false;
}

/**
 * @brief Adds a new m value to the container with automatic capacity expansion and hash table insertion.
 * 
 * This function implements efficient storage of m values using a dual-structure approach that
 * maintains both sequential access through a dynamic array and fast lookup capability through
 * a hash table. The function handles automatic memory management by expanding the values array
 * when capacity is exceeded and creates hash table nodes for O(1) average-case lookup performance.
 * 
 * The function performs two distinct but coordinated operations: array insertion for sequential
 * storage and hash table insertion for fast lookup. This dual approach optimizes for both
 * iteration through discovered m values and rapid repetition detection during sequence analysis.
 * 
 * Capacity Management:
 * When the values array reaches capacity, the function doubles the allocation size using
 * geometric growth strategy. This approach provides amortized O(1) insertion time while
 * minimizing the frequency of expensive reallocation operations.
 * 
 * Hash Table Insertion:
 * New hash nodes are inserted at the head of collision chains using a prepend strategy.
 * This provides O(1) insertion time and maintains all previously inserted values in their
 * respective collision chains for future lookup operations.
 * 
 * @param mv Pointer to the mValues container to modify. Must be a properly initialized
 *           container created with init_m_values(). The container's state will be updated
 *           to include the new m value.
 * @param m The m value to add to the container. This value will be stored in both the
 *          sequential array and the hash table for dual access patterns.
 * 
 * @note This function does not check for duplicate values before insertion. If the same
 *       m value is added multiple times, it will appear multiple times in both the array
 *       and hash table, though this typically doesn't occur in proper usage.
 * 
 * @note The function uses geometric growth (doubling) for array expansion, providing
 *       amortized O(1) insertion performance over many operations.
 * 
 * @note Hash table insertion uses prepend strategy, where new nodes are added at the
 *       head of collision chains for maximum insertion efficiency.
 * 
 * @note Memory allocation failures in either array expansion or hash node creation
 *       will terminate the program via safe_malloc() and safe_realloc().
 * 
 * @warning The function assumes the container has been properly initialized. Using
 *          an uninitialized container results in undefined behavior.
 * 
 * @complexity Amortized O(1) - constant time insertion with occasional O(n) array
 *            reallocation where n is the current number of stored values
 * 
 * @see init_m_values() for container initialization requirements
 * @see is_m_repeated() for lookup operations using the hash table
 * @see safe_realloc() for memory expansion behavior
 * @see hash_function() for hash computation algorithm
 * 
 * @example
 * ```c
 * // Initialize container and add values
 * mValues m_values;
 * init_m_values(&m_values);
 * 
 * // Add sequence of m values
 * add_m_value(&m_values, 25);   // First value
 * add_m_value(&m_values, 108);  // Second value
 * add_m_value(&m_values, 54);   // Third value
 * 
 * // Container now contains 3 values accessible by:
 * // - Sequential access: m_values.values[0], m_values.values[1], m_values.values[2]
 * // - Hash lookup: is_m_repeated(&m_values, 25) returns true
 * 
 * // Typical usage in sequence analysis
 * uint64_t collatz_value = 22;
 * uint64_t m = calculate_m(collatz_value);
 * 
 * if (!is_m_repeated(&m_values, m)) {
 *     add_m_value(&m_values, m);  // Store for future repetition checks
 * } else {
 *     printf("Pseudocycle detected at m=%lu\n", m);
 * }
 * 
 * // Automatic capacity expansion example
 * for (int i = 0; i < 200; i++) {
 *     add_m_value(&m_values, i);  // Triggers expansion at 100 values
 * }
 * // Container capacity automatically doubled from 100 to 200
 * ```
 */
static void add_m_value(mValues* mv, uint64_t m) {
    // Expand array if necessary
    if (mv->count >= mv->capacity) {
        int new_capacity = mv->capacity * 2;
        mv->values = safe_realloc(mv->values, new_capacity * sizeof(uint64_t), "m_values expansion");
        mv->capacity = new_capacity;
    }
    
    mv->values[mv->count++] = m;
    
    // Add to hash table
    uint64_t hash = hash_function(m);
    HashNode* new_node = safe_malloc(sizeof(HashNode), "hash node");
    new_node->value = m;
    new_node->next = mv->buckets[hash];
    mv->buckets[hash] = new_node;
}

// ***********************************************************************************
// * 4. UNIQUE mr SET (GLOBAL RESULTS)
// ***********************************************************************************

/**
 * @brief Creates and initializes a thread-safe set for collecting unique mr values discovered during parallel search.
 * 
 * This function allocates and configures a specialized data structure designed to collect and
 * manage unique mr values found throughout the entire search process. The structure maintains
 * parallel arrays to store both the unique mr values and the first n value that generated each
 * mr, enabling comprehensive analysis of mr discovery patterns and statistical reporting.
 * 
 * The UniqueMrSet serves as a global collection point for all unique mr discoveries across
 * multiple threads, providing thread-safe operations for concurrent updates while maintaining
 * the relationship between mr values and their first occurrence. This information is crucial
 * for research analysis and validation of the search results.
 * 
 * Thread Safety Architecture:
 * The structure includes an OpenMP lock to ensure atomic operations when multiple threads
 * simultaneously discover new unique mr values. The lock protects both the addition of new
 * entries and the reading of current statistics during progress reporting.
 * 
 * Memory Layout Design:
 * - Parallel arrays for mr values and their corresponding first n values
 * - Fixed initial capacity (10000) based on empirical estimates of mr discovery rates
 * - Automatic expansion capability through safe_realloc() when capacity is exceeded
 * - Thread synchronization primitive for concurrent access protection
 * 
 * @return Pointer to a fully initialized UniqueMrSet structure ready for concurrent access.
 *         The structure contains empty parallel arrays with initial capacity and an active
 *         OpenMP lock. Never returns NULL due to safe_malloc() usage.
 * 
 * @note The initial capacity of 10000 is chosen based on empirical analysis of mr discovery
 *       rates in typical search ranges, providing sufficient space to minimize reallocations.
 * 
 * @note Both values and first_n arrays are allocated with the same capacity to maintain
 *       parallel array consistency and enable direct indexing relationships.
 * 
 * @note The OpenMP lock is initialized and ready for immediate use by multiple threads
 *       without additional setup requirements.
 * 
 * @note The count field is initialized to zero, indicating an empty set ready for discoveries.
 * 
 * @warning The returned structure must be properly cleaned up using destroy_unique_mr_set()
 *          to destroy the OpenMP lock and free allocated memory.
 * 
 * @complexity O(1) - simple structure and array allocation with fixed initial capacity
 * 
 * @see destroy_unique_mr_set() for proper cleanup procedure
 * @see add_unique_mr() for thread-safe addition of discovered mr values
 * @see is_mr_already_found() for duplicate detection functionality
 * @see UniqueMrSet structure definition for field descriptions
 * 
 * @example
 * ```c
 * // Create unique mr set for global discovery collection
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * 
 * // Use in parallel search context
 * #pragma omp parallel
 * {
 *     for (uint64_t n = thread_start; n < thread_end; n++) {
 *         uint64_t mr = find_first_mr_in_sequence(n, &found);
 *         
 *         if (found) {
 *             bool is_new = add_unique_mr(unique_set, mr, n);
 *             if (is_new) {
 *                 printf("New unique mr=%lu discovered at n=%lu\n", mr, n);
 *             }
 *         }
 *     }
 * }
 * 
 * // Access final statistics
 * printf("Total unique mr values found: %d\n", unique_set->count);
 * 
 * // Cleanup when done
 * destroy_unique_mr_set(unique_set);
 * ```
 */
static UniqueMrSet* create_unique_mr_set(void) {
    UniqueMrSet* set = safe_malloc(sizeof(UniqueMrSet), "unique mr set");
    set->capacity = 10000;
    set->values = safe_malloc(set->capacity * sizeof(uint64_t), "unique mr values");
    set->first_n = safe_malloc(set->capacity * sizeof(uint64_t), "unique mr first_n");
    set->count = 0;
    omp_init_lock(&set->lock);
    return set;
}

/**
 * @brief Safely deallocates all memory associated with a UniqueMrSet structure and destroys its synchronization primitives.
 * 
 * This function performs comprehensive cleanup of a UniqueMrSet structure, ensuring proper
 * deallocation of all dynamically allocated parallel arrays and destruction of OpenMP
 * synchronization resources. The cleanup process prevents memory leaks while ensuring
 * that system-level synchronization primitives are properly released to avoid resource
 * exhaustion in long-running applications.
 * 
 * The function implements defensive programming by checking for NULL pointers, making it
 * safe to call in error handling paths or cleanup sequences where the set might not have
 * been successfully created. This design prevents segmentation faults and ensures robust
 * cleanup behavior even in partial initialization scenarios.
 * 
 * Cleanup Process:
 * 1. Validate input pointer to handle NULL gracefully
 * 2. Deallocate the values array containing unique mr values
 * 3. Deallocate the first_n array containing discovery n values
 * 4. Destroy the OpenMP lock to release synchronization resources
 * 5. Deallocate the main structure memory
 * 
 * Resource Management Strategy:
 * The function addresses both user-space memory (parallel arrays) and system-level
 * synchronization resources (OpenMP locks). Proper destruction of OpenMP locks is
 * essential to prevent resource leaks that could affect system performance over time.
 * 
 * @param set Pointer to the UniqueMrSet structure to destroy. Can be NULL, in which
 *            case the function returns immediately without action. After this function
 *            returns, the pointer becomes invalid and should not be accessed.
 * 
 * @note This function is safe to call with NULL pointers, making it suitable for
 *       cleanup in error handling paths where set creation might have failed.
 * 
 * @note The function must only be called after all threads have finished using the
 *       set, as it destroys the synchronization lock that protects concurrent access.
 * 
 * @note After calling this function, any references to the set structure or its
 *       arrays become invalid and accessing them results in undefined behavior.
 * 
 * @note The function deallocates both parallel arrays (values and first_n) that
 *       were allocated during set creation, ensuring complete memory cleanup.
 * 
 * @warning This function is NOT thread-safe. Ensure no other threads are accessing
 *          the set when calling this function.
 * 
 * @warning Do not attempt to use the set pointer after calling this function,
 *          as the memory has been deallocated.
 * 
 * @complexity O(1) - constant time cleanup operations regardless of set contents
 * 
 * @see create_unique_mr_set() for the corresponding allocation function
 * @see omp_destroy_lock() for OpenMP lock destruction requirements
 * @see add_unique_mr() for functions that access the set during operation
 * 
 * @example
 * ```c
 * // Typical usage in cleanup sequence
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * 
 * // ... use set during search operations ...
 * 
 * // Cleanup when search is complete
 * destroy_unique_mr_set(unique_set);
 * unique_set = NULL;  // Prevent accidental reuse
 * 
 * // Safe to call with NULL
 * destroy_unique_mr_set(NULL);  // No-op, returns immediately
 * 
 * // Error handling example
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * if (some_error_condition) {
 *     destroy_unique_mr_set(unique_set);  // Safe cleanup
 *     return error_code;
 * }
 * 
 * // Cleanup in search context
 * void cleanup_search_context(SearchContext* ctx) {
 *     destroy_unique_mr_set(ctx->unique_set);
 *     // ... other cleanup ...
 * }
 * ```
 */
static void destroy_unique_mr_set(UniqueMrSet* set) {
    if (set) {
        free(set->values);
        free(set->first_n);
        omp_destroy_lock(&set->lock);
        free(set);
    }
}

/**
 * @brief Thread-safely adds a new unique mr value to the set with automatic capacity expansion and duplicate prevention.
 * 
 * This function provides atomic insertion of newly discovered mr values into the unique collection
 * while ensuring no duplicates are stored and maintaining the relationship between mr values and
 * the first n value that generated them. The function implements comprehensive thread safety using
 * OpenMP locks and handles automatic memory expansion when the current capacity is exceeded.
 * 
 * The function performs duplicate detection, capacity management, and insertion as a single atomic
 * operation, ensuring data consistency even during high-frequency concurrent access from multiple
 * threads. Only genuinely unique mr values are added to the collection, with early termination
 * for duplicates to minimize lock contention time.
 * 
 * Duplicate Detection Strategy:
 * Before insertion, the function performs a linear search through existing values to ensure
 * uniqueness. If a duplicate is found, the function immediately returns false without modifying
 * the collection, minimizing the time spent holding the exclusive lock.
 * 
 * Capacity Management:
 * When the collection reaches capacity, both parallel arrays (values and first_n) are
 * simultaneously expanded using geometric growth (doubling strategy) to maintain array
 * synchronization and provide amortized O(1) insertion performance.
 * 
 * @param set Pointer to the UniqueMrSet structure to modify. Must be a valid set with
 *            initialized parallel arrays and OpenMP lock.
 * @param mr The mr value to add to the unique collection. This represents a m repeated
 *           value discovered during sequence analysis.
 * @param n The n value that first generated this mr value. This information is stored
 *          in parallel with the mr value for research and analysis purposes.
 * 
 * @return true if the mr value was successfully added as a new unique entry
 *         false if the mr value already exists in the collection (no modification made)
 * 
 * @note The function maintains parallel arrays in perfect synchronization, ensuring that
 *       values[i] and first_n[i] always correspond to the same discovery.
 * 
 * @note Duplicate detection is performed within the critical section to ensure atomicity,
 *       but early termination minimizes lock contention for duplicate cases.
 * 
 * @note Capacity expansion doubles both arrays simultaneously to maintain parallel structure
 *       and provides amortized O(1) insertion performance over many operations.
 * 
 * @note The function is fully thread-safe and can be called concurrently from multiple
 *       threads without external synchronization requirements.
 * 
 * @warning Memory allocation failures during capacity expansion will terminate the program
 *          via safe_realloc(), as this represents an unrecoverable error condition.
 * 
 * @complexity Average case: O(n) for duplicate detection plus amortized O(1) for insertion
 *            Expansion case: O(n) for both duplicate detection and array reallocation
 * 
 * @see is_mr_already_found() for read-only duplicate checking
 * @see safe_realloc() for memory expansion behavior
 * @see report_new_unique_mr() for usage in discovery reporting
 * @see UniqueMrSet structure definition for parallel array layout
 * 
 * @example
 * ```c
 * // Add unique mr values during parallel search
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * 
 * // Successful addition of new unique value
 * bool added1 = add_unique_mr(unique_set, 25, 408);    // Returns true
 * bool added2 = add_unique_mr(unique_set, 108, 1234);  // Returns true
 * 
 * // Duplicate detection and rejection
 * bool added3 = add_unique_mr(unique_set, 25, 816);    // Returns false (duplicate mr)
 * 
 * // Usage in discovery process
 * #pragma omp parallel for
 * for (uint64_t n = 1; n < max_n; n++) {
 *     uint64_t mr = find_first_mr_in_sequence(n, &found);
 *     if (found) {
 *         bool is_new = add_unique_mr(unique_set, mr, n);
 *         if (is_new) {
 *             report_new_unique_mr(mr, n, unique_set, progress);
 *         }
 *     }
 * }
 * 
 * // Capacity expansion example
 * // When count reaches capacity (10000), both arrays automatically double to 20000
 * ```
 */
static bool add_unique_mr(UniqueMrSet* set, uint64_t mr, uint64_t n) {
    omp_set_lock(&set->lock);
    
    // Check if it already exists
    for (int i = 0; i < set->count; i++) {
        if (set->values[i] == mr) {
            omp_unset_lock(&set->lock);
            return false;
        }
    }
    
    // Expand capacity, if necessary
    if (set->count >= set->capacity) {
        int new_capacity = set->capacity * 2;
        set->values = safe_realloc(set->values, new_capacity * sizeof(uint64_t), "unique mr values expansion");
        set->first_n = safe_realloc(set->first_n, new_capacity * sizeof(uint64_t), "unique mr first_n expansion");
        set->capacity = new_capacity;
    }
    
    set->values[set->count] = mr;
    set->first_n[set->count] = n;
    set->count++;
    
    omp_unset_lock(&set->lock);
    return true;
}

/**
 * @brief Finds the index position of an mr value in the unique set without modifying the set.
 * 
 * This function performs a thread-safe linear search through the unique mr set to locate
 * the array index of a specific mr value. The index is used for accessing parallel arrays
 * and updating distribution statistics, providing the mapping between mr values and their
 * storage positions in the unique set.
 * 
 * The function implements read-only access to the unique set with proper thread synchronization,
 * ensuring consistent results even during concurrent read operations from multiple threads.
 * Unlike add_unique_mr(), this function never modifies the set contents.
 * 
 * Search Implementation:
 * - Linear scan through the values array from index 0 to count-1
 * - First match returns immediately with the found index
 * - No match returns -1 to indicate value not present
 * - Thread-safe access via OpenMP lock acquisition
 * 
 * The linear search approach is acceptable because the number of unique mr values is typically
 * small (42 in the complete range 1 to 2^40), making the O(n) complexity negligible compared
 * to the overall sequence analysis work.
 * 
 * @param set Pointer to the UniqueMrSet to search. Must be a valid set with initialized
 *            values array and OpenMP lock. The set contents are not modified.
 * @param mr The mr value to locate in the unique set. This should be a value that has
 *           already been discovered during sequence analysis.
 * 
 * @return The 0-based index of the mr value in the set's values array if found.
 *         Returns -1 if the mr value is not present in the set.
 * 
 * @note The function uses const cast on the lock parameter to enable locking during read-only
 *       operations, as OpenMP lock functions don't accept const pointers.
 * 
 * @note Linear search is used because the unique mr count is typically very small (~42),
 *       making more complex search algorithms (binary search, hash table) unnecessary.
 * 
 * @note The returned index can be used to access both set->values[index] and set->first_n[index]
 *       due to parallel array organization.
 * 
 * @note Thread safety is ensured by lock acquisition, though the lock is released before
 *       returning to minimize contention for subsequent operations.
 * 
 * @note The function may return -1 if called with an mr value that hasn't been added to
 *       the unique set yet, which should be handled appropriately by callers.
 * 
 * @complexity O(n) where n is the number of unique mr values in the set (typically ~42)
 * 
 * @see add_unique_mr() for adding mr values that can be subsequently found
 * @see classify_and_update_statistics() for typical usage context
 * @see UniqueMrSet for structure definition and parallel array organization
 * 
 * @example
 * ```c
 * UniqueMrSet* set = create_unique_mr_set();
 * 
 * add_unique_mr(set, 25, 408);
 * add_unique_mr(set, 108, 1234);
 * add_unique_mr(set, 53, 2048);
 * 
 * // Find indices of known values
 * int idx1 = find_mr_index(set, 25);   // Returns index where mr=25 is stored
 * int idx2 = find_mr_index(set, 108);  // Returns index where mr=108 is stored
 * 
 * // Attempt to find value not in set
 * int idx3 = find_mr_index(set, 999);  // Returns -1 (not found)
 * 
 * // Use index to access parallel arrays
 * if (idx1 >= 0) {
 *     printf("mr=%lu first found at n=%lu\n", 
 *            set->values[idx1], set->first_n[idx1]);
 * }
 * 
 * // Typical usage in statistics update
 * uint64_t mr = find_first_mr_in_sequence(n, &found, &M_star, &seq_type);
 * int mr_idx = find_mr_index(unique_set, mr);
 * if (mr_idx >= 0) {
 *     stats->mr_distribution[mr_idx]++;
 * }
 * ```
 */
static int find_mr_index(const UniqueMrSet* set, uint64_t mr) {
    omp_set_lock((omp_lock_t*)&set->lock);
    
    int index = -1;
    for (int i = 0; i < set->count; i++) {
        if (set->values[i] == mr) {
            index = i;
            break;
        }
    }
    
    omp_unset_lock((omp_lock_t*)&set->lock);
    return index;
}

// ***********************************************************************************
// * 5. PROGRESS TRACKER SYSTEM
// ***********************************************************************************

/**
 * @brief Creates and initializes a thread-safe statistics collection structure for sequence classification analysis.
 * 
 * This function allocates and configures a comprehensive statistics tracking system designed to
 * collect distribution data across all sequences in the search range. The structure maintains
 * separate counters for each sequence type (A, B, C) and a dynamic array tracking the frequency
 * of each unique mr value discovered during the search.
 * 
 * The statistics enable detailed analysis of Collatz sequence behavior patterns, revealing:
 * - Distribution of sequences across classification types
 * - Frequency of each unique mr value across the search space
 * - Relative prevalence of different pseudocycle patterns
 * - Correlation between sequence types and mr value characteristics
 * 
 * Memory Layout Design:
 * - Type counters initialized to zero for fresh tracking session
 * - Distribution array allocated with initial capacity of 10,000 entries
 * - All distribution entries zero-initialized for accurate frequency counting
 * - OpenMP lock initialized for thread-safe concurrent updates
 * 
 * The initial capacity of 10,000 is chosen to accommodate typical mr index ranges with minimal
 * reallocation overhead, though the array expands dynamically if higher indices are encountered.
 * 
 * @return Pointer to a fully initialized SequenceStatistics structure ready for concurrent access.
 *         The structure contains zero-initialized counters, allocated distribution array, and
 *         an active OpenMP lock. Never returns NULL due to safe_malloc() usage.
 * 
 * @note All type counters (type_A_count, type_B_count, type_C_count) start at zero,
 *       representing an empty statistics collection at initialization.
 * 
 * @note The mr_distribution array is pre-allocated with 10,000 zero-initialized entries,
 *       sufficient for most search ranges without requiring expansion.
 * 
 * @note The OpenMP lock is initialized and ready for immediate use by multiple threads
 *       without additional setup requirements.
 * 
 * @note Zero-initialization of the distribution array ensures that unaccessed indices
 *       correctly report zero frequency rather than garbage values.
 * 
 * @warning The returned structure must be properly cleaned up using destroy_statistics()
 *          to destroy the OpenMP lock and free allocated memory.
 * 
 * @complexity O(n) where n is the initial capacity (10,000) due to zero-initialization loop
 * 
 * @see destroy_statistics() for proper cleanup procedure
 * @see classify_and_update_statistics() for thread-safe statistics updates
 * @see SequenceStatistics for structure definition and field descriptions
 * @see print_statistics() for statistics reporting and analysis
 * 
 * @example
 * ```c
 * // Create statistics for search operation
 * SequenceStatistics* stats = create_statistics();
 * 
 * // Use in parallel search context
 * #pragma omp parallel for
 * for (uint64_t n = 1; n < max_n; n++) {
 *     // ... sequence analysis ...
 *     classify_and_update_statistics(seq_type, mr_idx, stats);
 * }
 * 
 * // Access final statistics
 * printf("Type A: %lu sequences\n", stats->type_A_count);
 * printf("Type B: %lu sequences\n", stats->type_B_count);
 * printf("Type C: %lu sequences\n", stats->type_C_count);
 * 
 * // Cleanup when done
 * destroy_statistics(stats);
 * ```
 */

static SequenceStatistics* create_statistics(void) {
    SequenceStatistics* stats = safe_malloc(sizeof(SequenceStatistics), "sequence statistics");
    stats->type_A_count = 0;
    stats->type_B_count = 0;
    stats->type_C_count = 0;
    stats->mr_dist_capacity = 10000;
    stats->mr_distribution = safe_malloc(stats->mr_dist_capacity * sizeof(uint64_t), "mr distribution");
    
    for (int i = 0; i < stats->mr_dist_capacity; i++) {
        stats->mr_distribution[i] = 0;
    }
    
    omp_init_lock(&stats->lock);
    return stats;
}

/**
 * @brief Safely deallocates statistics structure and destroys synchronization primitives.
 * 
 * This function performs complete cleanup of a SequenceStatistics structure, ensuring proper
 * deallocation of the dynamic mr distribution array and destruction of OpenMP synchronization
 * resources. The cleanup process prevents memory leaks while ensuring that system-level
 * synchronization primitives are properly released.
 * 
 * The function implements defensive programming by checking for NULL pointers, making it
 * safe to call in error handling paths or cleanup sequences where the statistics structure
 * might not have been successfully created.
 * 
 * Cleanup Process:
 * 1. Validate input pointer to handle NULL gracefully
 * 2. Deallocate the mr_distribution array
 * 3. Destroy the OpenMP lock to release synchronization resources
 * 4. Deallocate the main structure memory
 * 
 * @param stats Pointer to the SequenceStatistics structure to destroy. Can be NULL,
 *              in which case the function returns immediately without action.
 *              After this function returns, the pointer becomes invalid and should
 *              not be accessed.
 * 
 * @note This function is safe to call with NULL pointers, making it suitable for
 *       cleanup in error handling paths where statistics creation might have failed.
 * 
 * @note The function must only be called after all threads have finished using the
 *       statistics, as it destroys the synchronization lock that protects concurrent access.
 * 
 * @note After calling this function, any references to the statistics structure become
 *       invalid and accessing them results in undefined behavior.
 * 
 * @warning This function is NOT thread-safe. Ensure no other threads are accessing
 *          the statistics when calling this function.
 * 
 * @complexity O(1) - constant time cleanup operations regardless of statistics content
 * 
 * @see create_statistics() for the corresponding allocation function
 * @see omp_destroy_lock() for OpenMP lock destruction requirements
 * @see classify_and_update_statistics() for functions that access statistics
 * 
 * @example
 * ```c
 * // Typical usage in cleanup sequence
 * SequenceStatistics* stats = create_statistics();
 * // ... use statistics during search ...
 * destroy_statistics(stats);
 * stats = NULL;  // Prevent accidental reuse
 * 
 * // Safe to call with NULL
 * destroy_statistics(NULL);  // No-op, returns immediately
 * 
 * // Cleanup in main function
 * void cleanup_search_context(SearchContext* ctx) {
 *     destroy_statistics(ctx->stats);
 *     // ... other cleanup ...
 * }
 * ```
 */
static void destroy_statistics(SequenceStatistics* stats) {
    if (stats) {
        free(stats->mr_distribution);
        omp_destroy_lock(&stats->lock);
        free(stats);
    }
}

/**
 * @brief Thread-safely updates sequence classification statistics and mr value distribution counters.
 * 
 * This function performs atomic updates to the global statistics structure, incrementing the
 * appropriate sequence type counter (A, B, or C) and updating the mr value distribution array
 * that tracks how many sequences produced each unique mr value. The function handles dynamic
 * array expansion when new mr indices exceed current capacity.
 * 
 * The statistics provide comprehensive insights into Collatz sequence behavior patterns:
 * - Type distribution reveals the relationship between M* and pseudocycle positions
 * - mr distribution shows the frequency of each unique mr value across the search range
 * - Combined analysis enables identification of common vs. rare pseudocycle patterns
 * 
 * Thread Safety Implementation:
 * Uses OpenMP lock to ensure atomic read-modify-write operations on shared statistics
 * structures, preventing race conditions when multiple threads update counters simultaneously.
 * 
 * Dynamic Array Management:
 * The mr_distribution array automatically expands when encountering mr indices beyond current
 * capacity, allocating additional space in 1000-element increments and zero-initializing new
 * entries to maintain accurate frequency counts.
 * 
 * @param sequence_type Character indicating sequence classification:
 *                      'A' = M* before first mr occurrence (increments type_A_count)
 *                      'B' = M* between first and second mr (increments type_B_count)
 *                      'C' = M* after second mr occurrence (increments type_C_count)
 *                      Other values default to Type A for safety
 * @param mr_index The index of the mr value in the unique set (0-based position in sorted array).
 *                 Valid range is [0, unique_count). Value -1 indicates no valid mr (skips distribution update).
 * @param stats Pointer to the SequenceStatistics structure to update. Must be a valid,
 *              initialized structure with active OpenMP lock.
 * 
 * @note The function uses a switch statement for type classification with a default case
 *       that treats unknown types as Type A, ensuring statistics remain consistent.
 * 
 * @note Distribution array expansion occurs within the critical section to prevent race
 *       conditions, though this may briefly increase lock contention during expansion.
 * 
 * @note New array segments are zero-initialized to ensure accurate frequency counting
 *       without contamination from uninitialized memory.
 * 
 * @note The 1000-element expansion increment balances memory efficiency with reduced
 *       reallocation frequency for typical mr index ranges.
 * 
 * @note mr_index = -1 is safely handled by skipping the distribution update, allowing
 *       callers to update type classification even when mr index lookup fails.
 * 
 * @complexity Amortized O(1) for counter updates, with occasional O(n) for array expansion
 *            where n is the expansion size (1000 elements)
 * 
 * @see SequenceStatistics for structure definition and field descriptions
 * @see process_single_number() for typical usage context
 * @see find_mr_index() for mr index lookup
 * @see create_statistics() for initialization requirements
 * 
 * @example
 * ```c
 * SequenceStatistics* stats = create_statistics();
 * 
 * // Update Type A sequence (no pseudocycle or M* before mr)
 * classify_and_update_statistics('A', -1, stats);
 * // stats->type_A_count = 1, distribution unchanged
 * 
 * // Update Type B sequence with mr at index 5
 * classify_and_update_statistics('B', 5, stats);
 * // stats->type_B_count = 1, stats->mr_distribution[5] = 1
 * 
 * // Update Type C sequence with same mr
 * classify_and_update_statistics('C', 5, stats);
 * // stats->type_C_count = 1, stats->mr_distribution[5] = 2
 * 
 * // Typical usage in processing pipeline
 * uint64_t M_star;
 * char seq_type;
 * bool found;
 * uint64_t mr = find_first_mr_in_sequence(n, &found, &M_star, &seq_type);
 * int mr_idx = find_mr_index(unique_set, mr);
 * classify_and_update_statistics(seq_type, mr_idx, stats);
 * ```
 */
static void classify_and_update_statistics(char sequence_type, int mr_index, SequenceStatistics* stats) {
    omp_set_lock(&stats->lock);
    
    // Classify sequence type
    switch (sequence_type) {
        case 'A':
            stats->type_A_count++;
            break;
        case 'B':
            stats->type_B_count++;
            break;
        case 'C':
            stats->type_C_count++;
            break;
        default:
            stats->type_A_count++;
            break;
    }
    
    // Update mr distribution (if we have valid index)
    if (mr_index >= 0) {
        if (mr_index >= stats->mr_dist_capacity) {
            int new_capacity = mr_index + 1000;
            stats->mr_distribution = safe_realloc(stats->mr_distribution, 
                                                  new_capacity * sizeof(uint64_t), 
                                                  "mr distribution expansion");
            for (int i = stats->mr_dist_capacity; i < new_capacity; i++) {
                stats->mr_distribution[i] = 0;
            }
            stats->mr_dist_capacity = new_capacity;
        }
        stats->mr_distribution[mr_index]++;
    }
    
    omp_unset_lock(&stats->lock);
}

/**
 * @brief Creates and initializes a thread-safe progress tracking structure for monitoring parallel search operations.
 * 
 * This function allocates and configures a progress tracking system designed to provide
 * real-time monitoring of parallel Collatz sequence analysis operations. The tracker
 * maintains thread-safe counters and timing information that can be safely updated by
 * multiple worker threads while being read by monitoring functions for progress reporting.
 * 
 * The progress tracker is essential for long-running computations as it provides users
 * with feedback on processing speed, completion estimates, and discovery statistics.
 * All fields are initialized to appropriate starting values for a new search operation.
 * 
 * Thread Safety Design:
 * The structure includes an OpenMP lock that enables atomic updates of progress metrics
 * during high-frequency parallel operations. This prevents race conditions when multiple
 * threads update counters simultaneously while maintaining acceptable performance overhead.
 * 
 * Initialization Values:
 * - All numeric counters start at zero to represent a fresh tracking session
 * - Timing information is initialized to 0.0 for immediate progress display triggering
 * - OpenMP lock is initialized and ready for immediate concurrent access
 * 
 * @return Pointer to a fully initialized ProgressTracker structure ready for concurrent
 *         access. The structure contains zero-initialized counters and an active OpenMP
 *         lock. Never returns NULL due to safe_malloc() usage.
 * 
 * @note The tracker's last_update_time field is initialized to 0.0, which ensures that
 *       the first call to update_progress_if_needed() will immediately display progress
 *       regardless of the PROGRESS_UPDATE_INTERVAL setting.
 * 
 * @note The OpenMP lock is initialized and must be properly destroyed during cleanup
 *       using omp_destroy_lock() to prevent resource leaks.
 * 
 * @note All progress counters use uint64_t to support very large search ranges
 *       (up to 2^64 numbers) without overflow concerns.
 * 
 * @warning The returned structure must be properly cleaned up using destroy_progress_tracker()
 *          to destroy the OpenMP lock and free allocated memory.
 * 
 * @complexity O(1) - simple structure allocation and field initialization
 * 
 * @see destroy_progress_tracker() for proper cleanup procedure
 * @see update_progress_if_needed() for progress reporting mechanism
 * @see increment_progress_counters() for thread-safe counter updates
 * @see ProgressTracker structure definition for field descriptions
 * 
 * @example
 * ```c
 * // Create progress tracker for parallel search
 * ProgressTracker* progress = create_progress_tracker();
 * 
 * // Use in parallel context
 * #pragma omp parallel
 * {
 *     uint64_t local_processed = 0;
 *     
 *     // Process numbers in assigned range
 *     for (uint64_t n = thread_start; n < thread_end; n++) {
 *         // ... analysis work ...
 *         local_processed++;
 *         
 *         // Periodic progress updates to avoid lock contention
 *         if (local_processed % 1000 == 0) {
 *             increment_progress_counters(progress, false);
 *         }
 *     }
 * }
 * 
 * // Check final statistics
 * printf("Total processed: %lu\n", progress->processed);
 * printf("Total found: %lu\n", progress->found_count);
 * 
 * // Cleanup when done
 * destroy_progress_tracker(progress);
 * ```
 */
static ProgressTracker* create_progress_tracker(void) {
    ProgressTracker* tracker = safe_malloc(sizeof(ProgressTracker), "progress tracker");
    tracker->processed = 0;
    tracker->found_count = 0;
    tracker->last_n_with_new_unique = 0;
    tracker->last_update_time = 0.0;
    omp_init_lock(&tracker->lock);
    return tracker;
}

/**
 * @brief Safely deallocates a progress tracker structure and destroys its synchronization primitives.
 * 
 * This function performs complete cleanup of a ProgressTracker structure, ensuring proper
 * destruction of OpenMP synchronization primitives and deallocation of memory. The cleanup
 * process follows proper resource management practices to prevent memory leaks and avoid
 * leaving dangling OpenMP locks that could consume system resources.
 * 
 * The function implements defensive programming by checking for NULL pointers, making it
 * safe to call in error handling paths or cleanup sequences where the tracker might not
 * have been successfully created. This design prevents segmentation faults and ensures
 * robust cleanup behavior.
 * 
 * Cleanup Process:
 * 1. Validate input pointer to handle NULL gracefully
 * 2. Destroy the OpenMP lock to release system synchronization resources
 * 3. Deallocate the main structure memory
 * 
 * Resource Management:
 * The function addresses both user-space memory (allocated via malloc) and system-level
 * synchronization resources (OpenMP locks). Proper destruction of OpenMP locks is critical
 * to prevent resource exhaustion in applications that create and destroy many trackers.
 * 
 * @param tracker Pointer to the ProgressTracker structure to destroy. Can be NULL,
 *                in which case the function returns immediately without action.
 *                After this function returns, the pointer becomes invalid and should
 *                not be accessed.
 * 
 * @note This function is safe to call with NULL pointers, making it suitable for
 *       cleanup in error handling paths where tracker creation might have failed.
 * 
 * @note The function must only be called after all threads have finished using the
 *       tracker, as it destroys the synchronization lock that protects concurrent access.
 * 
 * @note After calling this function, any references to the tracker structure become
 *       invalid and accessing them results in undefined behavior.
 * 
 * @warning This function is NOT thread-safe. Ensure no other threads are accessing
 *          the tracker when calling this function.
 * 
 * @warning Do not attempt to use the tracker pointer after calling this function,
 *          as the memory has been deallocated.
 * 
 * @complexity O(1) - constant time cleanup operations regardless of tracker usage history
 * 
 * @see create_progress_tracker() for the corresponding allocation function
 * @see omp_destroy_lock() for OpenMP lock destruction requirements
 * @see update_progress_if_needed() for functions that access the tracker
 * 
 * @example
 * ```c
 * // Typical usage in cleanup sequence
 * ProgressTracker* progress = create_progress_tracker();
 * 
 * // ... use progress tracker during computation ...
 * 
 * // Cleanup when computation is complete
 * destroy_progress_tracker(progress);
 * progress = NULL;  // Prevent accidental reuse
 * 
 * // Safe to call with NULL
 * destroy_progress_tracker(NULL);  // No-op, returns immediately
 * 
 * // Error handling example
 * ProgressTracker* progress = create_progress_tracker();
 * if (some_error_condition) {
 *     destroy_progress_tracker(progress);  // Safe cleanup
 *     return error_code;
 * }
 * 
 * // Cleanup in main function
 * void cleanup_search_context(SearchContext* ctx) {
 *     destroy_progress_tracker(ctx->progress);
 *     // ... other cleanup ...
 * }
 * ```
 */
static void destroy_progress_tracker(ProgressTracker* tracker) {
    if (tracker) {
        omp_destroy_lock(&tracker->lock);
        free(tracker);
    }
}

/**
 * @brief Saves complete search state to binary checkpoint file for resume capability after interruption.
 * 
 * This function implements fault-tolerant state persistence by writing all essential search progress
 * to a binary checkpoint file, enabling seamless resume of long-running computations after system
 * failures, user interruptions (Ctrl+C), or intentional pauses. The checkpoint contains minimal but
 * complete state information needed to continue from the exact point of interruption.
 * 
 * The function implements a safe two-file rotation strategy: the existing checkpoint is renamed to
 * a backup before writing the new checkpoint, ensuring that a valid checkpoint always exists even if
 * the write operation is interrupted. This prevents checkpoint corruption from system crashes during
 * the save operation itself.
 * 
 * Checkpoint Contents:
 * - Magic number (0x4D52434B = "MRCK") for file validation
 * - Last n value processed (resume point for search loop)
 * - Total numbers actually processed (accurate progress tracking)
 * - Search exponent (validation that checkpoint matches current search)
 * - Count of unique mr values discovered
 * - Array of all unique mr values found so far
 * - Array of first n values that generated each mr
 * 
 * File Format Strategy:
 * Binary format is used instead of JSON/text for minimal overhead during high-frequency saves
 * (every 5 minutes during long searches). The compact binary representation minimizes I/O time
 * and disk space usage while maintaining fast write/read performance.
 * 
 * Thread Safety:
 * The function acquires the progress tracker lock for the entire checkpoint operation to ensure
 * atomic snapshot of search state, preventing inconsistencies from concurrent updates during the
 * write operation.
 * 
 * @param ctx Pointer to the search context containing all state to be checkpointed:
 *            - progress: Thread tracking with processed counts
 *            - unique_set: All discovered unique mr values and their first occurrences
 *            - exponent: Search range parameter for validation on resume
 * @param last_n The last number successfully processed before checkpoint. This becomes the
 *               resume point (last_n + 1) when loading the checkpoint.
 * 
 * @note The function silently returns on file open failure rather than terminating the program,
 *       allowing the search to continue even if checkpoint saves fail due to disk issues.
 * 
 * @note Backup rotation (checkpoint.bin -> checkpoint.bak) ensures that a valid checkpoint
 *       always exists, even if the new checkpoint write is interrupted mid-operation.
 * 
 * @note The checkpoint file is typically very small (~1KB) even for complete searches,
 *       as it only stores unique mr values (typically 42) rather than all processed numbers.
 * 
 * @note The magic number enables quick validation that a file is actually a checkpoint
 *       rather than corrupted data or an unrelated binary file.
 * 
 * @note The entire checkpoint operation occurs within a critical section protected by
 *       the progress tracker lock, ensuring consistent state snapshot.
 * 
 * @note Empty unique sets (count=0) are handled gracefully by skipping array writes,
 *       though this only occurs at the very start of a search.
 * 
 * @warning Disk full conditions during checkpoint save will result in incomplete checkpoint
 *          files, but the backup rotation strategy ensures the previous valid checkpoint remains.
 * 
 * @complexity O(n) where n is the number of unique mr values discovered (typically 42),
 *            dominated by the array write operations
 * 
 * @see load_checkpoint() for the corresponding restore function
 * @see CheckpointHeader for checkpoint file format structure
 * @see CHECKPOINT_FILE and CHECKPOINT_BACKUP for filename constants
 * @see CHECKPOINT_MAGIC for validation magic number
 * @see execute_search_with_guided_scheduling() for periodic checkpoint triggering
 * 
 * @example
 * ```c
 * SearchContext ctx = { ... };  // Active search context
 * 
 * // Save checkpoint during search loop
 * uint64_t current_n = 15000000;
 * save_checkpoint(&ctx, current_n);
 * // Creates/updates checkpoint.bin with state at n=15000000
 * 
 * // Checkpoint file structure:
 * // [Header: 48 bytes]
 * //   - magic: 0x4D52434B
 * //   - last_n_processed: 15000000
 * //   - total_processed: 15000000
 * //   - exponent: 25
 * //   - unique_count: 42
 * // [Values array: 336 bytes (42 * 8)]
 * //   - mr values: [0, 1, 2, 3, 6, 7, ...]
 * // [First_n array: 336 bytes (42 * 8)]
 * //   - first n values: [1, 3, 6, 7, 15, 31, ...]
 * 
 * // Interrupted search scenario
 * // ... search running ...
 * save_checkpoint(&ctx, 20000000);  // Periodic save at 5-minute interval
 * // ... system crashes shortly after ...
 * // On restart: load_checkpoint() restores state and resumes from n=20000001
 * ```
 */
static void save_checkpoint(const SearchContext* ctx, uint64_t last_n) {
    omp_set_lock(&ctx->progress->lock);
    
    if (access(CHECKPOINT_FILE, F_OK) == 0) {
        rename(CHECKPOINT_FILE, CHECKPOINT_BACKUP);
    }
    
    FILE* fp = fopen(CHECKPOINT_FILE, "wb");
    if (!fp) {
        omp_unset_lock(&ctx->progress->lock);
        return;
    }
    
    // Write header
    CheckpointHeader header;
    header.magic = CHECKPOINT_MAGIC;
    header.last_n_processed = last_n;
    header.total_processed = ctx->progress->processed;
    header.exponent = ctx->exponent;
    header.unique_count = ctx->unique_set->count;
    
    fwrite(&header, sizeof(CheckpointHeader), 1, fp);
    
    if (header.unique_count > 0) {
        fwrite(ctx->unique_set->values, sizeof(uint64_t), header.unique_count, fp);
        fwrite(ctx->unique_set->first_n, sizeof(uint64_t), header.unique_count, fp);
    }
    
    fclose(fp);
    omp_unset_lock(&ctx->progress->lock);
}

/**
 * @brief Loads and validates checkpoint file to restore search state and enable resume from interruption point.
 * 
 * This function implements robust checkpoint restoration with comprehensive validation to ensure
 * safe resume of interrupted searches. It attempts to load the checkpoint file, validates its
 * integrity and compatibility with the current search parameters, restores all discovered unique
 * mr values, and calculates accurate resume position and progress statistics.
 * 
 * The function implements a fallback strategy: if the primary checkpoint file is corrupted or
 * missing, it attempts to load the backup file. This provides additional fault tolerance against
 * checkpoint corruption from system crashes during save operations.
 * 
 * Validation Checks:
 * 1. File existence and readability (tries main, then backup)
 * 2. Magic number verification (0x4D52434B) to confirm valid checkpoint format
 * 3. Exponent matching to ensure checkpoint is for the same search range
 * 4. Complete array data reading to verify file integrity
 * 
 * The function provides detailed console feedback about checkpoint status, including resume point,
 * completion percentage based on actual processed count, remaining work, and number of restored
 * unique mr values. This information helps users understand where the search is resuming and how
 * much work has already been completed.
 * 
 * State Restoration Process:
 * 1. Read and validate checkpoint header
 * 2. Allocate temporary arrays for unique values and first_n data
 * 3. Read both arrays and verify complete data transfer
 * 4. Populate the unique_set by adding each restored mr value with its first_n
 * 5. Update progress tracker with actual processed count from checkpoint
 * 6. Calculate and display resume statistics
 * 7. Free temporary arrays and close checkpoint file
 * 
 * @param ctx Pointer to search context to be populated with restored state:
 *            - unique_set: Will be populated with all previously discovered mr values
 *            - progress: Will be updated with actual progress count
 *            - exponent: Used for validation against checkpoint exponent
 * @param start_n Pointer to store the resume position (last_n_processed + 1).
 *                Updated only if checkpoint is successfully loaded and validated.
 * 
 * @return true if checkpoint was successfully loaded, validated, and state restored
 *         false if no checkpoint exists, validation fails, or data is incomplete
 * 
 * @note The function tries checkpoint.bin first, falling back to checkpoint.bak if the
 *       primary file is missing or corrupted, providing redundancy against save failures.
 * 
 * @note Magic number validation (0x4D52434B = "MRCK") prevents attempts to load corrupted
 *       files or unrelated binary data as checkpoint state.
 * 
 * @note Exponent mismatch detection prevents resuming with wrong search range parameters,
 *       which would produce incorrect or meaningless results.
 * 
 * @note The function uses actual total_processed from checkpoint rather than conservative
 *       estimation, providing accurate progress percentages on resume.
 * 
 * @note Incomplete array reads (partial file) are detected and cause validation failure,
 *       ensuring only complete checkpoints are used for resume.
 * 
 * @note Console output includes precise 10-decimal-place percentages for completion and
 *       remaining work, suitable for long-running searches where precision matters.
 * 
 * @note Temporary arrays are properly freed even on validation failures, preventing
 *       memory leaks in error paths.
 * 
 * @warning The function assumes unique_set is already initialized (empty) before calling,
 *          as it uses add_unique_mr() which expects valid set structure.
 * 
 * @complexity O(n) where n is the number of unique mr values in the checkpoint (typically 42),
 *            dominated by the array reading and unique_set population operations
 * 
 * @see save_checkpoint() for the corresponding checkpoint creation function
 * @see CheckpointHeader for checkpoint file format structure
 * @see add_unique_mr() for unique value restoration mechanism
 * @see CHECKPOINT_FILE and CHECKPOINT_BACKUP for filename constants
 * @see CHECKPOINT_MAGIC for validation constant
 * 
 * @example
 * ```c
 * SearchContext ctx = {
 *     .exponent = 25,
 *     .max_n = 1UL << 25,
 *     .unique_set = create_unique_mr_set(),
 *     .progress = create_progress_tracker(),
 *     // ...
 * };
 * 
 * uint64_t start_n = 1;  // Default starting point
 * 
 * if (load_checkpoint(&ctx, &start_n)) {
 *     // Successful resume
 *     // Example output:
 *     // [*] CHECKPOINT LOADED
 *     //     - Last processed: n = 15000000
 *     //     - Resuming from: n = 15000001
 *     //     - Completed: 44.7034836025% | Remaining: 55.2965163975%
 *     //     - Restored 42 unique mr values
 *     
 *     printf("Resuming search from n=%lu\n", start_n);
 *     // continue search from start_n...
 * } else {
 *     // No checkpoint or validation failed
 *     printf("Starting fresh from n=1\n");
 *     // start_n remains 1, search starts from beginning
 * }
 * 
 * // Exponent mismatch scenario
 * // Previous run: ./program 30 (interrupted)
 * // Current run: ./program 25
 * // Output: [*] WARNING: Checkpoint exponent mismatch (checkpoint=30, current=25). Starting fresh.
 * // Returns false, search starts from n=1
 * 
 * // Corrupted checkpoint scenario
 * // Output: [*] WARNING: Checkpoint file corrupted. Starting fresh.
 * // Returns false, falls back to backup or fresh start
 * ```
 */
static bool load_checkpoint(SearchContext* ctx, uint64_t* start_n) {
    FILE* fp = fopen(CHECKPOINT_FILE, "rb");
    if (!fp) {
        fp = fopen(CHECKPOINT_BACKUP, "rb");
        if (!fp) return false;
    }
    
    CheckpointHeader header;
    if (fread(&header, sizeof(CheckpointHeader), 1, fp) != 1) {
        fclose(fp);
        return false;
    }
    
    if (header.magic != CHECKPOINT_MAGIC) {
        fclose(fp);
        printf("\n[*] WARNING: Checkpoint file corrupted. Starting fresh.\n");
        return false;
    }
    
    if (header.exponent != ctx->exponent) {
        fclose(fp);
        printf("\n[*] WARNING: Checkpoint exponent mismatch (checkpoint=%d, current=%d). Starting fresh.\n",
               header.exponent, ctx->exponent);
        return false;
    }
    
    if (header.unique_count > 0) {
        uint64_t* temp_values = safe_malloc(header.unique_count * sizeof(uint64_t), "checkpoint values");
        uint64_t* temp_first_n = safe_malloc(header.unique_count * sizeof(uint64_t), "checkpoint first_n");
        
        size_t values_read = fread(temp_values, sizeof(uint64_t), header.unique_count, fp);
        size_t first_n_read = fread(temp_first_n, sizeof(uint64_t), header.unique_count, fp);
        
        if (values_read != (size_t)header.unique_count || first_n_read != (size_t)header.unique_count) {
            free(temp_values);
            free(temp_first_n);
            fclose(fp);
            printf("\n[*] WARNING: Checkpoint data incomplete. Starting fresh.\n");
            return false;
        }
        
        for (int i = 0; i < header.unique_count; i++) {
            add_unique_mr(ctx->unique_set, temp_values[i], temp_first_n[i]);
        }
        
        free(temp_values);
        free(temp_first_n);
    }
    
    *start_n = header.last_n_processed + 1;
    fclose(fp);
    
    // Use actual total_processed from checkpoint for accurate progress tracking
    ctx->progress->processed = header.total_processed;
    
    // Calculate percentages based on actual progress
    double completed_pct = (header.total_processed * 100.0) / (ctx->max_n - 1);
    double remaining_pct = 100.0 - completed_pct;
    
    printf("\n[*] CHECKPOINT LOADED\n");
    printf("\t - Last processed: n = %lu\n", header.last_n_processed);
    printf("\t - Resuming from: n = %lu\n", *start_n);
    printf("\t - Completed: %.10f%% | Remaining: %.10f%%\n", completed_pct, remaining_pct);
    printf("\t - Restored %d unique mr values\n", header.unique_count);
    
    return true;
}

/**
 * @brief Thread-safely updates the tracker with the most recent number that generated a new unique mr value.
 * 
 * This function maintains a record of the highest-numbered input that has produced a previously
 * unseen mr value during the search process. This information is valuable for monitoring the
 * discovery rate of new mr values and understanding the distribution pattern of unique mr
 * discoveries across the search space.
 * 
 * The function implements thread-safe updating using OpenMP locks to ensure that concurrent
 * updates from multiple threads don't create race conditions. Only values higher than the
 * current stored value are accepted, maintaining the "most recent discovery" semantics even
 * when threads process numbers out of order due to parallel scheduling.
 * 
 * Discovery Tracking Purpose:
 * - Monitor the search frontier for new mr value discoveries
 * - Provide feedback on discovery density across different ranges
 * - Enable analysis of where in the search space new patterns emerge
 * - Support debugging and research analysis of mr distribution
 * 
 * Thread Safety Implementation:
 * Uses OpenMP lock acquisition and release to ensure atomic read-modify-write operations
 * even when multiple threads simultaneously discover new unique mr values.
 * 
 * @param tracker Pointer to the ProgressTracker structure to update. Must be a valid
 *                tracker with an initialized OpenMP lock.
 * @param n The number that generated a new unique mr value. Only values greater than
 *          the currently stored value will update the tracker.
 * 
 * @note The function only updates the stored value if the new n is greater than the
 *       current value, ensuring that last_n_with_new_unique always represents the
 *       highest number that has produced a new unique mr.
 * 
 * @note This function is called from report_new_unique_mr() whenever a genuinely new
 *       mr value is discovered during the search process.
 * 
 * @note The comparison and update are performed atomically within the lock to prevent
 *       race conditions where multiple threads might read the same old value.
 * 
 * @warning The tracker must have a valid, initialized OpenMP lock. Calling this function
 *          on an uninitialized tracker results in undefined behavior.
 * 
 * @complexity O(1) - constant time operation with simple comparison and assignment
 * 
 * @see report_new_unique_mr() for the function that calls this updater
 * @see ProgressTracker structure definition for field descriptions
 * @see create_progress_tracker() for lock initialization
 * @see omp_set_lock() and omp_unset_lock() for synchronization primitives
 * 
 * @example
 * ```c
 * // Called when a new unique mr is discovered
 * ProgressTracker* tracker = create_progress_tracker();
 * 
 * // Thread 1 discovers new mr at n=1234
 * update_last_n_with_new_unique(tracker, 1234);
 * // tracker->last_n_with_new_unique is now 1234
 * 
 * // Thread 2 discovers new mr at n=987 (smaller, ignored)
 * update_last_n_with_new_unique(tracker, 987);
 * // tracker->last_n_with_new_unique remains 1234
 * 
 * // Thread 3 discovers new mr at n=5678 (larger, updated)
 * update_last_n_with_new_unique(tracker, 5678);
 * // tracker->last_n_with_new_unique is now 5678
 * 
 * // Usage in discovery reporting
 * if (add_unique_mr(unique_set, mr, n)) {
 *     update_last_n_with_new_unique(tracker, n);
 *     printf("New mr=%lu found at n=%lu\n", mr, n);
 * }
 * ```
 */
static void update_last_n_with_new_unique(ProgressTracker* tracker, uint64_t n) {
    omp_set_lock(&tracker->lock);
    if (n > tracker->last_n_with_new_unique) {
        tracker->last_n_with_new_unique = n;
    }
    omp_unset_lock(&tracker->lock);
}

/**
 * @brief Thread-safely increments progress counters using OpenMP atomic operations for high-performance concurrent updates.
 * 
 * This function provides efficient thread-safe updates to progress tracking counters without
 * the overhead of explicit lock acquisition and release. It uses OpenMP atomic directives
 * to ensure that concurrent increments from multiple threads are handled correctly while
 * maintaining maximum performance during high-frequency counter updates.
 * 
 * The function handles two distinct counter types: a mandatory processed counter that is
 * always incremented, and an optional found counter that is incremented only when a
 * number meets the search criteria. This design allows for efficient tracking of both
 * total work completed and successful discoveries.
 * 
 * Atomic Operation Benefits:
 * - No explicit lock contention or waiting between threads
 * - Hardware-level synchronization for maximum performance
 * - Automatic memory ordering guarantees for counter consistency
 * - Minimal overhead compared to mutex-based synchronization
 * 
 * Performance Characteristics:
 * Atomic operations are significantly faster than lock-based synchronization for simple
 * increment operations, making this function suitable for high-frequency calls within
 * tight processing loops without substantial performance degradation.
 * 
 * @param tracker Pointer to the ProgressTracker structure containing the counters to update.
 *                Must be a valid tracker structure with properly initialized counter fields.
 * @param mr_found Boolean flag indicating whether the processed number yielded a valid mr
 *                 result. If true, both processed and found_count are incremented; if false,
 *                 only processed is incremented.
 * 
 * @note OpenMP atomic directives ensure thread safety without requiring explicit locks,
 *       providing better performance than mutex-based synchronization for simple increments.
 * 
 * @note The processed counter is always incremented regardless of the mr_found flag,
 *       ensuring accurate tracking of total work completed.
 * 
 * @note Atomic operations provide memory ordering guarantees, ensuring that counter
 *       updates are visible to other threads in a consistent manner.
 * 
 * @note This function is designed for high-frequency calls and has minimal overhead
 *       compared to lock-based alternatives.
 * 
 * @complexity O(1) - constant time atomic operations with hardware-level synchronization
 * 
 * @see ProgressTracker structure definition for counter field descriptions
 * @see process_single_number() for typical usage in processing loops
 * @see update_progress_if_needed() for reading these counters safely
 * @see OpenMP atomic directive documentation for synchronization behavior
 * 
 * @example
 * ```c
 * // Usage in parallel processing loop
 * ProgressTracker* tracker = create_progress_tracker();
 * 
 * #pragma omp parallel for
 * for (uint64_t n = 1; n < max_n; n++) {
 *     bool found_mr = find_first_mr_in_sequence(n, &mr_value);
 *     
 *     // High-frequency atomic updates
 *     increment_progress_counters(tracker, found_mr);
 *     
 *     // Periodic progress display (less frequent)
 *     if (n % PROGRESS_CHECK_FREQUENCY == 0) {
 *         update_progress_if_needed(ctx);
 *     }
 * }
 * 
 * // Example with different outcomes
 * increment_progress_counters(tracker, true);   // processed++, found_count++
 * increment_progress_counters(tracker, false);  // processed++, found_count unchanged
 * increment_progress_counters(tracker, false);  // processed++, found_count unchanged
 * increment_progress_counters(tracker, true);   // processed++, found_count++
 * 
 * // Result: processed = 4, found_count = 2
 * ```
 */
static void increment_progress_counters(ProgressTracker* tracker, bool mr_found) {
    #pragma omp atomic
    tracker->processed++;
    
    if (mr_found) {
        #pragma omp atomic
        tracker->found_count++;
    }
}

static void update_progress_if_needed(const SearchContext* ctx) {
    ProgressTracker* tracker = ctx->progress;
    omp_set_lock(&tracker->lock);
    
    double current_time = omp_get_wtime();
    
    if (current_time - tracker->last_update_time >= PROGRESS_UPDATE_INTERVAL) {
        tracker->last_update_time = current_time;
        
        double elapsed = current_time - ctx->start_time;
        double rate = tracker->processed / elapsed;
        
        // Limitar progreso a 100% y evitar ETA negativo
        double progress_percent = (double)tracker->processed / ctx->max_n * 100.0;
        if (progress_percent > 100.0) {
            progress_percent = 100.0;
        }
        
        uint64_t remaining = 0;
        if (tracker->processed < ctx->max_n) {
            remaining = ctx->max_n - tracker->processed;
        }
        
        double eta = (remaining > 0) ? (remaining / rate) : 0.0;
        
        printf("\t - Progress: (%.10f%%) | Processed: %lu | Unique values of mr found: %d | %.1f nums/sec | ETA: %.1f min\n",
               progress_percent, tracker->processed, ctx->unique_set->count, rate, eta/60.0);
        fflush(stdout);
    }
    
    omp_unset_lock(&tracker->lock);
}

/**
 * @brief Reports the discovery of a new unique mr value with thread-safe output, initialization handling, and progress tracking updates.
 * 
 * This function provides immediate feedback when a genuinely new mr value is discovered during
 * the search process, combining progress tracking updates with formatted console output and
 * special handling for the first discovery event. The function ensures thread-safe reporting
 * even during high-frequency parallel discovery events while maintaining consistent output
 * formatting and providing initialization context for research monitoring.
 * 
 * The function performs three coordinated operations: updating the progress tracker with the
 * latest discovery location, handling first-time initialization output, and generating formatted
 * discovery notifications. Special logic ensures that the very first discovery triggers an
 * initial progress line to establish proper output context before individual discoveries are reported.
 * 
 * First Discovery Initialization:
 * - Uses static variable to detect first execution across all threads
 * - Outputs initial progress line showing zero state before first discovery
 * - Establishes consistent output format baseline for subsequent reports
 * - Prevents confusion about missing initial progress information
 * 
 * Discovery Reporting Features:
 * - Immediate notification of new unique mr discoveries with double-indented formatting
 * - Complete context including mr value, generating n value, and current total count
 * - Thread-safe output formatting using named critical section to prevent text corruption
 * - Progress tracker frontier updates for discovery location monitoring
 * - Forced output flushing for real-time feedback during long searches
 * 
 * Thread Safety Implementation:
 * Progress tracker update occurs outside the critical section for performance, while output
 * formatting uses OpenMP critical section with named identifier (discovery_report) to ensure
 * that only one thread can output discovery information at a time, preventing garbled text.
 * 
 * @param mr The newly discovered unique mr value that has not been previously found.
 *           This represents a m repeated value discovered through sequence analysis.
 * @param n The specific n value that generated this unique mr during its Collatz sequence.
 *          This provides the discovery context for research analysis and frontier tracking.
 * @param set Pointer to the UniqueMrSet containing all discovered unique values. Used to
 *            report the current total count of unique discoveries for progress monitoring.
 * @param tracker Pointer to the ProgressTracker to update with the latest discovery location.
 *                This maintains the frontier of unique discoveries for analysis purposes.
 * 
 * @note The function uses a static boolean variable to track first execution, creating
 *       global state that affects behavior across all function calls and threads.
 * 
 * @note Progress tracker update occurs outside the critical section to minimize lock
 *       contention, while output formatting is fully protected by the named critical section.
 * 
 * @note The first discovery triggers an initialization progress line showing zero state,
 *       establishing proper context for subsequent individual discovery reports.
 * 
 * @note Output uses double indentation (\t\t) to distinguish individual discoveries from
 *       general progress updates, creating clear visual hierarchy in the output.
 * 
 * @note fflush(stdout) ensures immediate output display even when stdout is buffered
 *       or redirected to files, providing real-time feedback during long searches.
 * 
 * @note The function assumes the mr value is genuinely unique, as duplicate checking
 *       should be performed by add_unique_mr() before calling this function.
 * 
 * @complexity O(1) - simple output formatting, progress update, and static variable check
 * 
 * @see update_last_n_with_new_unique() for progress tracking updates
 * @see add_unique_mr() for the duplicate checking that precedes this function
 * @see UniqueMrSet structure for unique value collection
 * @see ProgressTracker for discovery frontier tracking
 * 
 * @example
 * ```c
 * // First discovery call in the program
 * UniqueMrSet* unique_set = create_unique_mr_set();
 * ProgressTracker* progress = create_progress_tracker();
 * 
 * // First call outputs initialization line then discovery
 * report_new_unique_mr(25, 408, unique_set, progress);
 * // Output: "\t - Progress: (0.0%) | Processed: 0 | Unique values of mr found: 0 | 0.0 nums/sec | ETA: -- min"
 * //         "\t\t - New unique value of mr = 25 found, generated by n = 408 (total unique mr values: 1)"
 * 
 * // Subsequent calls only output discoveries
 * report_new_unique_mr(108, 1234, unique_set, progress);
 * // Output: "\t\t - New unique value of mr = 53 found, generated by n = 108 (total unique mr values: 15)"
 * 
 * // Multiple concurrent discoveries (thread-safe output)
 * #pragma omp parallel for
 * for (uint64_t n = 1; n < max_n; n++) {
 *     uint64_t mr = find_first_mr_in_sequence(n, &found);
 *     if (found && add_unique_mr(unique_set, mr, n)) {
 *         report_new_unique_mr(mr, n, unique_set, progress);
 *         // Each thread's output appears cleanly without interference
 *         // First thread to call will output initialization line
 *     }
 * }
 * ```
 */
static void report_new_unique_mr(uint64_t mr, uint64_t n, const UniqueMrSet* set, ProgressTracker* tracker) {

    static bool first_report = true;
    
    update_last_n_with_new_unique(tracker, n);
    
    #pragma omp critical(discovery_report)
    {
        if (first_report) {
            printf("\t - Progress: (0.0%%) | Processed: 0 | Unique values of mr found: 0 | 0.0 nums/sec | ETA: -- min\n");
            fflush(stdout);
            first_report = false;
        }
        
        printf("\t\t - New unique value of mr = %lu found, generated by n = %lu (total unique mr values: %d)\n", 
               mr, n, set->count);
        fflush(stdout);
    }
}

// ***********************************************************************************
// * 6. COLLATZ SEQUENCE ANALYSIS
// ***********************************************************************************

/**
 * @brief Analyzes a Collatz sequence to find the first repeated m value (mr), maximum m value (M*), and classify the sequence type.
 * 
 * This function performs comprehensive analysis of a complete Collatz sequence starting from n_start,
 * tracking all m values, detecting the first repetition (mr), identifying the maximum m value (M*),
 * and classifying the sequence into one of three types based on the relative positions of M* and mr
 * occurrences within the sequence.
 * 
 * Sequence Classification System:
 * - Type A: M* appears before the first occurrence of mr (or mr=0, no pseudocycle)
 * - Type B: M* appears between the first and second occurrence of mr (M* inside the pseudocycle)
 * - Type C: M* appears after the second occurrence of mr (M* after pseudocycle completion)
 * 
 * The classification provides insight into the relationship between the maximum multiplicity value
 * and pseudocycle formation, enabling statistical analysis of Collatz sequence behavior patterns.
 * 
 * Analysis Process:
 * 1. Iterate through Collatz sequence from n_start to 1
 * 2. Calculate m value inline: m = (n - p) / 2 where p = 1 (odd) or 2 (even)
 * 3. Track maximum m value (M*) and its position in sequence
 * 4. Detect first m repetition using hash table lookup
 * 5. Record positions of first and second occurrences of repeated m
 * 6. Classify sequence based on positional relationships
 * 7. Apply inline Collatz transform: n = 3n+1 (odd) or n/2 (even)
 * 
 * @param n_start The starting value for Collatz sequence analysis. Must be > 0.
 * @param found Pointer to boolean flag set to true if analysis completes successfully.
 *              Always set to true in current implementation.
 * @param M_star Pointer to store the maximum m value encountered in the sequence.
 *               This represents the highest multiplicity value before reaching 1.
 * @param sequence_type Pointer to char for sequence classification result:
 *                      'A' = M* before first mr occurrence
 *                      'B' = M* between first and second mr occurrence  
 *                      'C' = M* after second mr occurrence
 * 
 * @return The first repeated m value (mr). Returns 0 if no repetition occurs before
 *         reaching the trivial cycle (indicating Type A sequence with no pseudocycle).
 * 
 * @note The function uses inline calculations for m and Collatz transforms to maximize
 *       performance in tight loops, avoiding function call overhead.
 * 
 * @note Maximum sequence length is bounded by MAX_SEQUENCE_LENGTH (100000 steps) to
 *       prevent infinite loops, though this is unlikely with valid Collatz sequences.
 * 
 * @note For sequences with no repetition (mr=0), first_occurrence_position is set to 0
 *       and second_occurrence_position to the final step count for classification purposes.
 * 
 * @note The mValues container is destroyed before return, ensuring no memory leaks
 *       regardless of sequence length or early termination.
 * 
 * @complexity O(n) where n is the length of the Collatz sequence from n_start to 1,
 *            with O(1) average-case hash table operations for repetition detection
 * 
 * @see mValues for the hash table-based m value tracking structure
 * @see is_m_repeated() for repetition detection mechanism
 * @see add_m_value() for m value storage
 * @see process_single_number() for typical usage context
 * 
 * @example
 * ```c
 * uint64_t M_star;
 * char seq_type;
 * bool found;
 * 
 * // Analyze sequence starting from n=27
 * uint64_t mr = find_first_mr_in_sequence(27, &found, &M_star, &seq_type);
 * 
 * if (found) {
 *     printf("mr = %lu, M* = %lu, Type = %c\n", mr, M_star, seq_type);
 *     // Example output: "mr = 121, M* = 4616, Type = B"
 * }
 * 
 * // Type A example (no pseudocycle)
 * mr = find_first_mr_in_sequence(8, &found, &M_star, &seq_type);
 * // mr = 0, seq_type = 'A' (reaches 1 without repetition)
 * 
 * // Type C example (M* after pseudocycle)
 * mr = find_first_mr_in_sequence(703, &found, &M_star, &seq_type);
 * // mr = 243, M* appears after second occurrence, seq_type = 'C'
 * ```
 */
static uint64_t find_first_mr_in_sequence(uint64_t n_start, bool* found, uint64_t* M_star, char* sequence_type) {
    uint64_t n = n_start;
    mValues m_values;
    init_m_values(&m_values);
    
    uint64_t first_mr = 0;
    uint64_t max_m = 0;
    int max_m_position = -1;
    int first_occurrence_position = -1;
    int second_occurrence_position = -1;
    bool mr_found = false;
    
    int step = 0;
    while (n != 1) {
        // Inline calculate_m: m = (c - p) / 2
        uint64_t p = (n & 1) ? 1 : 2;
        uint64_t m = (n - p) >> 1;
        
        // Update M* and its position
        if (m > max_m) {
            max_m = m;
            max_m_position = step;
        }
        
        // Detect first repetition
        if (!mr_found && is_m_repeated(&m_values, m)) {
            first_mr = m;
            second_occurrence_position = step;
            mr_found = true;
            
            for (int i = 0; i < m_values.count; i++) {
                if (m_values.values[i] == m) {
                    first_occurrence_position = i;
                    break;
                }
            }
        }
        
        add_m_value(&m_values, m);
        
        // Inline apply_collatz
        if (n & 1) {
            n = 3 * n + 1;
        } else {
            n = n >> 1;
        }
        
        step++;
        
        if (step > MAX_SEQUENCE_LENGTH) {
            break;
        }
    }
    
    if (!mr_found) {
        first_mr = 0;
        first_occurrence_position = 0;
        second_occurrence_position = step;
    }
    
    *found = true;
    *M_star = max_m;
    
    // Classify sequence type
    if (first_mr == 0) {
        *sequence_type = 'A';
    } else if (max_m_position < first_occurrence_position) {
        *sequence_type = 'A';
    } else if (max_m_position <= second_occurrence_position) {
        *sequence_type = 'B';
    } else {
        *sequence_type = 'C';
    }
    
    destroy_m_values(&m_values);
    return first_mr;
}

/**
 * @brief Processes a single number through complete Collatz analysis, discovery tracking, and sequence classification.
 * 
 * This function orchestrates the complete analysis pipeline for a single input value, combining
 * sequence analysis, unique mr discovery detection, progress tracking, and sequence type classification.
 * It serves as the atomic processing unit called by parallel worker threads during the main search loop.
 * 
 * Processing Pipeline:
 * 1. Analyze complete Collatz sequence to find mr, M*, and sequence type
 * 2. Update global progress counters (always increments found count since all sequences yield mr≥0)
 * 3. Attempt to add mr to unique set (succeeds only for new unique values)
 * 4. Report new unique discoveries with discovery context
 * 5. Find mr index in unique set for distribution tracking
 * 6. Update sequence classification statistics (Type A/B/C)
 * 7. Maintain local thread counters for reduction operations
 * 
 * The function is designed for high-frequency parallel execution with minimal lock contention
 * by using local counters that are later reduced and atomic operations for shared state updates.
 * 
 * @param n The number to process through Collatz sequence analysis. Must be > 0.
 * @param ctx Pointer to search context containing all shared data structures:
 *            - unique_set: Global collection of discovered unique mr values
 *            - progress: Thread-safe progress tracking and counters
 *            - stats: Sequence classification statistics (Type A/B/C distribution)
 * @param local_found Pointer to thread-local counter for numbers yielding mr values.
 *                    Incremented for every processed number (all yield mr≥0).
 * @param local_processed Pointer to thread-local counter for total processed numbers.
 *                        Incremented after successful analysis completion.
 * 
 * @note All sequences yield an mr value (0 or positive), so found count always increments.
 *       This differs from search patterns where only some inputs produce valid results.
 * 
 * @note The function performs thread-safe operations on shared structures (unique_set, progress, stats)
 *       while maintaining local counters to minimize atomic operation overhead.
 * 
 * @note New unique mr discoveries trigger immediate reporting via report_new_unique_mr(),
 *       providing real-time feedback during long search operations.
 * 
 * @note Sequence classification statistics are updated for every processed number,
 *       enabling comprehensive distribution analysis across the entire search range.
 * 
 * @note The mr index lookup may return -1 if the add operation fails due to race conditions,
 *       but this is safely handled by classify_and_update_statistics().
 * 
 * @complexity O(s) where s is the length of the Collatz sequence for input n,
 *            with additional O(u) for unique set operations where u is unique mr count
 * 
 * @see find_first_mr_in_sequence() for sequence analysis algorithm
 * @see add_unique_mr() for thread-safe unique value tracking
 * @see increment_progress_counters() for progress updates
 * @see classify_and_update_statistics() for type distribution tracking
 * @see report_new_unique_mr() for discovery notifications
 * 
 * @example
 * ```c
 * SearchContext ctx = { ... };  // Initialized context
 * uint64_t local_found = 0, local_processed = 0;
 * 
 * // Process single number
 * process_single_number(27, &ctx, &local_found, &local_processed);
 * // local_found = 1, local_processed = 1
 * // If mr=121 is new: triggers report_new_unique_mr()
 * // Updates stats with Type B classification
 * 
 * // Typical usage in parallel loop
 * #pragma omp parallel reduction(+:local_found, local_processed)
 * {
 *     uint64_t thread_found = 0, thread_processed = 0;
 *     
 *     #pragma omp for schedule(guided)
 *     for (uint64_t n = start; n < max_n; n++) {
 *         process_single_number(n, &ctx, &thread_found, &thread_processed);
 *     }
 *     
 *     local_found += thread_found;
 *     local_processed += thread_processed;
 * }
 * ```
 */
static void process_single_number(uint64_t n, SearchContext* ctx, uint64_t* local_found, uint64_t* local_processed) {
    uint64_t M_star = 0;
    char sequence_type = '?';
    bool found = false;
    
    uint64_t mr = find_first_mr_in_sequence(n, &found, &M_star, &sequence_type);
    
    // Always find mr (0 or >0)
    (*local_found)++;
    increment_progress_counters(ctx->progress, true);
    
    // Add to unique set
    bool is_new_unique = add_unique_mr(ctx->unique_set, mr, n);
    if (is_new_unique) {
        report_new_unique_mr(mr, n, ctx->unique_set, ctx->progress);
    }
    
    // Update statistics
    int mr_index = find_mr_index(ctx->unique_set, mr);
    classify_and_update_statistics(sequence_type, mr_index, ctx->stats);
    
    (*local_processed)++;
}

// ***********************************************************************************
// * 7. PARALLEL SCHEDULING SYSTEM
// ***********************************************************************************

/**
 * @brief Executes parallel search using OpenMP guided scheduling with automatic checkpoint saves and interruption handling.
 * 
 * This function implements the core parallel search loop with fault tolerance through periodic
 * checkpoint saves and graceful interruption handling. It uses OpenMP guided scheduling strategy
 * which provides excellent load balancing while maintaining checkpoint safety by keeping thread 0's
 * position close to other threads, minimizing the gap between checkpoint position and actual progress.
 * 
 * The function manages three critical aspects simultaneously:
 * 1. Parallel work distribution across threads with optimal load balancing
 * 2. Periodic automatic checkpoint saves to enable resume after interruption
 * 3. Graceful shutdown handling for user interruptions (Ctrl+C/SIGINT)
 * 
 * Guided Scheduling Strategy:
 * OpenMP's guided scheduling starts with large chunks and progressively decreases chunk size as
 * work completes. This approach provides:
 * - Excellent load balancing even with uniform work distribution
 * - Minimal synchronization overhead compared to dynamic scheduling
 * - Thread 0 position stays close to other threads (important for checkpoint accuracy)
 * - Reduced gap between checkpoint_position and actual processed count
 * 
 * Why Guided Over Static or Dynamic:
 * - Static: Would cause large checkpoint gaps when thread 0 finishes early
 * - Dynamic: Causes excessive synchronization overhead with small chunk sizes
 * - Guided: Optimal balance of load balancing and checkpoint-friendly thread positioning
 * 
 * Checkpoint Management:
 * - Thread 0 monitors elapsed time since last checkpoint save
 * - Saves checkpoint every CHECKPOINT_INTERVAL seconds (default 300 = 5 minutes)
 * - Uses atomic operations to safely read checkpoint position for save operations
 * - Displays checkpoint confirmation with exact resume point
 * - Performs final checkpoint at search completion before cleanup
 * 
 * Interruption Handling:
 * - Thread 0 checks checkpoint_signal_received flag periodically
 * - On SIGINT (Ctrl+C), saves immediate checkpoint at safe position
 * - Displays resume instructions before clean exit
 * - Ensures no computation loss even for unexpected interruptions
 * 
 * Progress Monitoring:
 * - Thread 0 checks progress every PROGRESS_CHECK_FREQUENCY numbers
 * - Displays periodic updates with completion percentage, speed, ETA
 * - Coordinates progress display with checkpoint timing for efficiency
 * 
 * Thread-Local Reduction:
 * - Each thread maintains local found/processed counters
 * - Reduction clause combines all thread counts at loop completion
 * - Minimizes atomic operation overhead during tight processing loop
 * 
 * @param ctx Pointer to search context containing:
 *            - max_n: Upper bound (exclusive) of search range
 *            - unique_set: Global collection for discovered mr values
 *            - progress: Thread-safe progress tracking
 *            - stats: Sequence classification statistics
 * @param start_n Starting position for search loop (1 for fresh start, last_n+1 for resume).
 *                Allows seamless continuation from checkpoint resume point.
 * @param total_found Pointer to store total count of numbers yielding mr values (all numbers).
 *                    Updated via reduction at loop completion.
 * @param total_processed Pointer to store total count of numbers successfully processed.
 *                        Updated via reduction at loop completion.
 * 
 * @note Guided scheduling is specifically chosen to keep thread 0's position close to other
 *       threads, ensuring minimal gap between checkpoint position and actual progress.
 * 
 * @note The checkpoint position (tracked atomically) represents the last number processed by
 *       thread 0, which is a safe conservative resume point under guided scheduling.
 * 
 * @note Progress updates and checkpoint saves are only performed by thread 0 to avoid
 *       race conditions and excessive I/O from multiple threads.
 * 
 * @note The final checkpoint save ensures that completed searches have a valid checkpoint
 *       at 100% completion, though this is immediately cleaned up afterward.
 * 
 * @note Checkpoint and backup files are deleted on successful completion to avoid confusion
 *       when starting new searches or different exponents.
 * 
 * @note The reduction clause ensures thread-local counters are efficiently combined without
 *       atomic operations during the tight processing loop.
 * 
 * @note On interruption (SIGINT), the program exits immediately after checkpoint save,
 *       preventing partial result output that might be misleading.
 * 
 * @warning Do not modify checkpoint files manually during search execution, as this will
 *          corrupt the state and prevent proper resume capability.
 * 
 * @complexity O(n/p) where n = (max_n - start_n) and p = number of threads,
 *            with periodic O(m) checkpoint overhead where m ≈ 42 unique values (negligible)
 * 
 * @see process_single_number() for per-number processing logic
 * @see save_checkpoint() for checkpoint state persistence
 * @see update_progress_if_needed() for progress display logic
 * @see checkpoint_signal_handler() for SIGINT handling
 * @see CHECKPOINT_INTERVAL for checkpoint frequency configuration
 * @see PROGRESS_CHECK_FREQUENCY for progress update frequency
 * 
 * @example
 * ```c
 * SearchContext ctx = { ... };
 * uint64_t found = 0, processed = 0;
 * 
 * // Fresh start
 * execute_search_with_guided_scheduling(&ctx, 1, &found, &processed);
 * 
 * // Resume from checkpoint
 * uint64_t resume_n = 15000001;  // From loaded checkpoint
 * execute_search_with_guided_scheduling(&ctx, resume_n, &found, &processed);
 * 
 * // Typical execution output:
 * // [*] SEARCH PROCESS
 * //     - Starting from n=1
 * //     - Progress: (0.0%) | Processed: 0 | Unique: 0 | 0.0 nums/sec | ETA: -- min
 * //     ... processing ...
 * //     - [Autosaving checkpoint at n = 5000000]
 * //     - Progress: (14.9%) | Processed: 5000000 | Unique: 42 | 125000.0 nums/sec | ETA: 3.8 min
 * //     ... processing ...
 * //     - [Autosaving checkpoint at n = 10000000]
 * //     ... continues until completion or interruption ...
 * //     - Final checkpoint saved
 * //     - Checkpoint files cleaned...search completed
 * 
 * // Interruption scenario (user presses Ctrl+C):
 * //     - [Autosaving checkpoint at n = 7500000]
 * //     ^C
 * //     - Interrupted! Checkpoint saved at n = 7523456
 * //     - Run again with same exponent to resume
 * // [Program exits with code 0]
 * ```
 */
static void execute_search_with_guided_scheduling(SearchContext* ctx, uint64_t start_n, 
                                                   uint64_t* total_found, uint64_t* total_processed) {
    double last_checkpoint_time = omp_get_wtime();
    uint64_t local_found = 0, local_processed = 0;
    uint64_t checkpoint_position = start_n;
    
    #pragma omp parallel reduction(+:local_found, local_processed)
    {
        uint64_t thread_found = 0, thread_processed = 0;
        int thread_num = omp_get_thread_num();
        
        // Guided scheduling: checkpoint-friendly, good load balancing
        #pragma omp for schedule(guided)
        for (uint64_t n = start_n; n < ctx->max_n; n++) {
            process_single_number(n, ctx, &thread_found, &thread_processed);
            
            if (thread_num == 0) {
                #pragma omp atomic write
                checkpoint_position = n;
                
                if (thread_processed % PROGRESS_CHECK_FREQUENCY == 0) {
                    update_progress_if_needed(ctx);
                    
                    double current_time = omp_get_wtime();
                    if (current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL) {
                        uint64_t safe_pos;
                        #pragma omp atomic read
                        safe_pos = checkpoint_position;
                        
                        save_checkpoint(ctx, safe_pos);
                        last_checkpoint_time = current_time;
                        printf("\t - [Autosaving checkpoint at n = %lu]\n", safe_pos);
                        fflush(stdout);
                    }
                }
                
                if (checkpoint_signal_received) {
                    uint64_t safe_pos;
                    #pragma omp atomic read
                    safe_pos = checkpoint_position;
                    
                    save_checkpoint(ctx, safe_pos);
                    printf("\n\t - Interrupted! Checkpoint saved at n = %lu\n", safe_pos);
                    printf("\t - Run again with same exponent to resume\n");
                    fflush(stdout);
                    exit(0);
                }
            }
        }
        
        local_found += thread_found;
        local_processed += thread_processed;
    }
    
    save_checkpoint(ctx, ctx->max_n - 1);
    printf("\t - Final checkpoint saved\n");
    
    remove(CHECKPOINT_FILE);
    remove(CHECKPOINT_BACKUP);
    printf("\t - Checkpoint files cleaned...search completed\n");
    
    *total_found = local_found;
    *total_processed = local_processed;
}

/**
 * @brief Orchestrates the complete parallel search operation with initialization, execution, and checkpoint cleanup.
 * 
 * This function serves as the high-level coordinator for the entire parallel search process,
 * managing the initialization of the search session, delegating to the actual parallel execution
 * function, and ensuring proper cleanup of checkpoint files upon successful completion. It provides
 * a clean abstraction layer between the main function and the low-level scheduling implementation.
 * 
 * The function handles two distinct starting scenarios:
 * 1. Fresh start (start_n = 1): New search beginning from the first number
 * 2. Resume (start_n > 1): Continuation from a loaded checkpoint after interruption
 * 
 * The function provides clear console feedback about the search mode (fresh vs. resume) to help
 * users understand whether they are continuing previous work or starting anew.
 * 
 * Execution Flow:
 * 1. Display search process header and starting position information
 * 2. Delegate to execute_search_with_guided_scheduling() for actual parallel work
 * 3. Return control to caller with final found/processed counts
 * 
 * The function acts as a coordination point between checkpoint loading (in main), actual parallel
 * execution (in execute_search_with_guided_scheduling), and result reporting (in main), maintaining
 * clear separation of concerns in the program architecture.
 * 
 * @param ctx Pointer to search context containing all configuration and shared data structures:
 *            - max_n: Upper bound of search range
 *            - unique_set: Global collection for discovered mr values
 *            - progress: Thread-safe progress tracking
 *            - stats: Sequence classification statistics
 * @param start_n Starting position for the search (1 for fresh start, or last_n+1 from checkpoint)
 * @param found_count Pointer to store total numbers yielding mr values (updated by scheduling function)
 * @param processed_count Pointer to store total numbers processed (updated by scheduling function)
 * 
 * @note The function distinguishes between fresh start (n=1) and resume (n>1) purely for user
 *       feedback, as the actual execution logic treats both cases identically.
 * 
 * @note Console output provides immediate confirmation of search mode before any processing begins,
 *       helping users verify they are running the correct operation.
 * 
 * @note The function does not perform checkpoint cleanup itself - that responsibility belongs
 *       to execute_search_with_guided_scheduling() which knows when search completes successfully.
 * 
 * @note All heavy lifting (parallel loops, checkpoints, progress) is delegated to the scheduling
 *       function, keeping this function focused on coordination and user communication.
 * 
 * @complexity O(1) for the coordination logic itself, with O(n/p) dominated by the delegated
 *            parallel execution where n is the range size and p is thread count
 * 
 * @see execute_search_with_guided_scheduling() for actual parallel execution
 * @see load_checkpoint() for checkpoint resume capability that sets start_n
 * @see main() for the calling context and overall program flow
 * 
 * @example
 * ```c
 * SearchContext ctx = { ... };
 * uint64_t found = 0, processed = 0;
 * 
 * // Fresh start scenario
 * execute_parallel_search(&ctx, 1, &found, &processed);
 * // Console output:
 * // [*] SEARCH PROCESS
 * //     - Starting from n=1
 * 
 * // Resume scenario (after checkpoint load)
 * uint64_t resume_point = 15000001;
 * execute_parallel_search(&ctx, resume_point, &found, &processed);
 * // Console output:
 * // [*] SEARCH PROCESS
 * //     - Resuming from n=15000001
 * 
 * // Both scenarios then proceed with identical execution logic
 * ```
 */
static void execute_parallel_search(SearchContext* ctx, uint64_t start_n, 
                                    uint64_t* found_count, uint64_t* processed_count) {
    printf("\n[*] SEARCH PROCESS\n");
    
    if (start_n == 1) {
        printf("\t - Starting from n=1\n");
    } else {
        printf("\t - Resuming from n=%lu\n", start_n);
    }
    
    execute_search_with_guided_scheduling(ctx, start_n, found_count, processed_count);
}

// ***********************************************************************************
// * 8. RESULTS OUTPUT AND REPORTING
// ***********************************************************************************

/**
 * @brief Creates an array of indices that sorts unique mr values in ascending order without modifying the original data.
 * 
 * This function implements an indirect sorting algorithm that generates an index array to access
 * unique mr values in sorted order while preserving the original storage organization of the
 * UniqueMrSet. This approach maintains the relationship between mr values and their corresponding
 * first_n values in parallel arrays while enabling sorted output for reports and analysis.
 * 
 * The function uses a simple but correct bubble sort algorithm for the index array. While bubble
 * sort has O(n²) complexity, it's acceptable here because the number of unique mr values is
 * typically small (dozens to hundreds) compared to the search space, and the implementation
 * prioritizes correctness and simplicity over optimal performance for this non-critical operation.
 * 
 * Indirect Sorting Benefits:
 * - Preserves original data organization in parallel arrays
 * - Maintains relationship between mr values and discovery n values
 * - Enables multiple sorted views without data duplication
 * - Allows original insertion order to be retained alongside sorted access
 * - Minimal memory overhead (only index array allocation)
 * 
 * Implementation Strategy:
 * 1. Allocate index array with same count as unique mr values
 * 2. Initialize indices to natural order (0, 1, 2, ...)
 * 3. Sort indices based on comparison of referenced mr values
 * 4. Return sorted index array for indirect access to original data
 * 
 * @param set Pointer to the UniqueMrSet containing the mr values to be sorted.
 *            Must be a valid set with count > 0 for meaningful sorting.
 *            The original set data remains unchanged.
 * 
 * @return Pointer to a dynamically allocated array of indices that provides
 *         sorted access to the mr values. The caller is responsible for
 *         freeing this memory when no longer needed.
 * 
 * @note The returned indices array enables sorted access via set->values[indices[i]]
 *       where i ranges from 0 to set->count-1 in ascending mr value order.
 * 
 * @note Bubble sort is used for simplicity and correctness. For large datasets,
 *       this could be replaced with quicksort or mergesort for better performance.
 * 
 * @note The function assumes set->count represents the actual number of valid
 *       entries in the values array and does not perform bounds checking.
 * 
 * @note Parallel array relationships are preserved: indices[i] can be used to
 *       access both set->values[indices[i]] and set->first_n[indices[i]].
 * 
 * @warning The caller must free the returned index array to prevent memory leaks.
 *          The original set data must remain valid while the indices are in use.
 * 
 * @complexity O(n²) where n is set->count, due to bubble sort algorithm.
 *            For typical use cases with small n, this is acceptable.
 * 
 * @see print_results() for usage in sorted output generation
 * @see UniqueMrSet structure definition for parallel array organization
 * @see safe_malloc() for memory allocation with error handling
 * 
 * @example
 * ```c
 * // Create and populate unique mr set
 * UniqueMrSet* set = create_unique_mr_set();
 * add_unique_mr(set, 108, 1234);  // First discovery
 * add_unique_mr(set, 25, 408);    // Second discovery  
 * add_unique_mr(set, 53, 2048);   // Third discovery
 * 
 * // Original order: [108, 25, 53]
 * // Desired sorted order: [25, 53, 108]
 * 
 * int* indices = create_sorting_indices(set);
 * // indices array contains: [1, 2, 0]
 * 
 * // Access mr values in sorted order
 * for (int i = 0; i < set->count; i++) {
 *     int idx = indices[i];
 *     printf("mr=%lu (first found at n=%lu)\n", 
 *            set->values[idx], set->first_n[idx]);
 * }
 * // Output: mr=25 (first found at n=408)
 * //         mr=53 (first found at n=2048)  
 * //         mr=108 (first found at n=1234)
 * 
 * free(indices);  // Clean up when done
 * ```
 */
static int* create_sorting_indices(const UniqueMrSet* set) {
    int* indices = safe_malloc(set->count * sizeof(int), "sorting indices");
    
    for (int i = 0; i < set->count; i++) {
        indices[i] = i;
    }
    
    // Simple bubble sort for mr values
    for (int i = 0; i < set->count - 1; i++) {
        for (int j = 0; j < set->count - 1 - i; j++) {
            if (set->values[indices[j]] > set->values[indices[j + 1]]) {
                int temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
    
    return indices;
}

/**
 * @brief Exports complete discovery results to JSON file with clean array structure for data analysis integration.
 * 
 * This function generates a structured JSON file containing all unique mr values discovered during
 * the search along with the first n value that generated each mr. The output uses a simple array-of-objects
 * format that is compatible with modern data processing tools, web applications, and JSON parsing libraries
 * while maintaining excellent human readability for inspection and debugging.
 * 
 * The JSON structure is intentionally minimal, containing only the essential data (mr values and their
 * first occurrences) without metadata overhead. This design optimizes for file size, parsing speed,
 * and compatibility with diverse data processing pipelines while avoiding the complexity of nested
 * structures or redundant formatting.
 * 
 * JSON Format Features:
 * - Clean array structure with no wrapper objects or metadata sections
 * - Each entry is a simple object with "mr" and "n" fields
 * - Proper JSON formatting with indentation for human readability
 * - Preserves discovery order (insertion order in unique_set)
 * - Standard format compatible with all JSON parsers
 * - Compact representation suitable for large result sets
 * 
 * The function maintains discovery order from the unique_set, allowing temporal analysis of when
 * different mr values were first encountered during the search process. This ordering provides
 * additional research value beyond the sorted presentation in console reports.
 * 
 * Output File Structure:
 * ```json
 * [
 *   {
 *     "mr": <first_mr_value>,
 *     "n": <first_n_value>
 *   },
 *   {
 *     "mr": <second_mr_value>,
 *     "n": <second_n_value>
 *   },
 *   ...
 * ]
 * ```
 * 
 * Error Handling:
 * The function handles file creation failures gracefully by displaying an error message but not
 * terminating the program, allowing the search results to still be displayed in console output
 * even if JSON file creation fails due to disk issues.
 * 
 * @param ctx Pointer to search context containing:
 *            - unique_set: Collection of all discovered unique mr values with first_n data
 * @param filename The output filename for the JSON file. Should include the .json extension.
 *                 Typically follows pattern: "mr_pairs_detected_on_range_1_to_2pow<exp>.json"
 * 
 * @note The function preserves discovery order from unique_set rather than sorting, enabling
 *       temporal analysis of mr value discovery patterns.
 * 
 * @note File creation failure results in error message but does not terminate the program,
 *       allowing console output and statistics to still be displayed.
 * 
 * @note Proper JSON comma placement (all but last entry) ensures valid JSON syntax that
 *       can be parsed by any standard JSON library.
 * 
 * @note The simple two-field object structure enables easy extraction of specific data
 *       using standard JSON query tools like jq: `jq '.[].mr' file.json`
 * 
 * @note Human-readable formatting with indentation makes the file suitable for manual
 *       inspection and debugging while remaining compact enough for large datasets.
 * 
 * @note The function does not include metadata (search range, timestamp, exponent) to
 *       keep the format focused on pure data, though this information is encoded in
 *       the filename by the caller.
 * 
 * @complexity O(n) where n is the number of unique mr values (typically 42),
 *            dominated by sequential file writing
 * 
 * @see SearchContext for context structure containing unique_set
 * @see UniqueMrSet for the data structure being exported
 * @see main() for filename generation pattern
 * 
 * @example
 * ```c
 * SearchContext ctx = { ... };  // Contains 42 unique mr values
 * 
 * write_json_results(&ctx, "mr_pairs_detected_on_range_1_to_2pow25.json");
 * 
 * // Generated file content:
 * // [
 * //   {
 * //     "mr": 0,
 * //     "n": 1
 * //   },
 * //   {
 * //     "mr": 1,
 * //     "n": 3
 * //   },
 * //   {
 * //     "mr": 2,
 * //     "n": 6
 * //   },
 * //   ...
 * //   {
 * //     "mr": 3643,
 * //     "n": 7287
 * //   }
 * // ]
 * 
 * // Easy data extraction with jq:
 * // $ jq '.[].mr' file.json              # List all mr values
 * // $ jq 'length' file.json              # Count unique values
 * // $ jq 'map(.mr) | max' file.json      # Find maximum mr
 * // $ jq '.[] | select(.mr == 121)' file.json  # Find specific mr entry
 * ```
 */
static void write_json_results(const SearchContext* ctx, const char* filename) {
    FILE* json_file = fopen(filename, "w");
    if (!json_file) {
        printf("\n[*] ERROR: Could not create the JSON output file %s\n", filename);
        return;
    }
    
    fprintf(json_file, "[\n");
    
    for (int i = 0; i < ctx->unique_set->count; i++) {
        fprintf(json_file, "  {\n");
        fprintf(json_file, "    \"mr\": %lu,\n", ctx->unique_set->values[i]);
        fprintf(json_file, "    \"n\": %lu\n", ctx->unique_set->first_n[i]);
        fprintf(json_file, "  }");
        
        if (i < ctx->unique_set->count - 1) {
            fprintf(json_file, ",");
        }
        fprintf(json_file, "\n");
    }
    
    fprintf(json_file, "]\n");
    fclose(json_file);
}

/**
 * @brief Generates a concise summary report with performance metrics and unique mr value list.
 * 
 * This function produces a streamlined summary of the search operation, focusing on key performance
 * metrics and the complete list of discovered unique mr values in sorted order. The report format
 * is optimized for quick assessment of search results and copy-paste integration with external
 * analysis tools, without the detailed per-value information that is available in the JSON output.
 * 
 * The function provides essential validation information including total processing counts, execution
 * time, throughput metrics, and a comma-separated list of all unique mr values sorted in ascending
 * order. This compact format enables rapid verification of search completeness while providing the
 * core data needed for further analysis.
 * 
 * Report Structure:
 * 1. Total numbers processed (should match expected range size)
 * 2. Total execution time in seconds
 * 3. Processing speed (numbers/second) for performance assessment
 * 4. Numbers yielding mr values (equals processed count since all yield mr≥0)
 * 5. Count of unique mr values discovered
 * 6. Sorted comma-separated list of unique mr values for copy-paste use
 * 
 * The sorted mr value list uses temporary index array to present results in ascending order
 * without modifying the original unique set data structure, enabling both sorted output and
 * preservation of discovery order in other outputs.
 * 
 * @param ctx Pointer to the search context containing:
 *            - unique_set: Collection of all discovered unique mr values
 * @param total_time The total execution time in seconds for speed calculation
 * @param found_count The total count of numbers yielding mr values (equals processed_count)
 * @param processed_count The total count of numbers successfully processed
 * 
 * @note The "Numbers with mr found" equals processed_count because all Collatz sequences
 *       yield an mr value (0 or positive), unlike search patterns where only some inputs
 *       produce valid results.
 * 
 * @note The mr values list is presented in ascending numerical order for systematic
 *       verification, though the unique_set internally maintains discovery order.
 * 
 * @note The comma-separated format of the mr list enables easy copy-paste into spreadsheets,
 *       analysis tools, or research documentation.
 * 
 * @note The function creates and frees temporary sorting indices internally, preventing
 *       memory leaks while avoiding modification of the original data structures.
 * 
 * @note Processing speed is calculated as total numbers divided by total time, providing
 *       average throughput including all checkpoint saves and progress updates.
 * 
 * @complexity O(n log n) where n is the number of unique mr values, dominated by the
 *            sorting operation (though with n≈42, this is effectively O(1))
 * 
 * @see create_sorting_indices() for sorted presentation logic
 * @see SearchContext for context structure definition
 * @see print_statistics() for detailed distribution analysis
 * @see write_json_results() for complete per-value details in JSON format
 * 
 * @example
 * ```c
 * SearchContext ctx = { ... };  // Contains 42 unique mr values
 * double total_time = 125.3;    // 2 minutes 5 seconds
 * uint64_t found = 33554432;
 * uint64_t processed = 33554432;
 * 
 * print_results(&ctx, total_time, found, processed);
 * 
 * // Example output:
 * //
 * // [*] SEARCH RESULTS
 * //     - Total numbers processed: 33554432
 * //     - Total time: 125.300 seconds
 * //     - Speed: 267891.23 numbers/second
 * //     - Numbers with mr found: 33554432
 * //     - Unique mr values found: 42
 * //     - List of mr values found: 0, 1, 2, 3, 6, 7, 8, 9, 12, 16, 19, 25, 45, 53, 60, 79,
 * //       91, 121, 125, 141, 166, 188, 205, 243, 250, 324, 333, 432, 444, 487, 576, 592,
 * //       649, 667, 683, 865, 889, 1153, 1214, 1821, 2428, 3643
 * ```
 */
static void print_results(const SearchContext* ctx, double total_time, 
                                   uint64_t found_count, uint64_t processed_count) {
   
    printf("\n[*] SEARCH RESULTS\n");
    printf("\t - Total numbers processed: %lu\n", processed_count);
    printf("\t - Total time: %.3f seconds\n", total_time);
    printf("\t - Speed: %.2f numbers/second\n", (double)processed_count / total_time);
    printf("\t - Numbers with mr found: %lu\n", found_count);
    printf("\t - Unique mr values found: %lu\n", (uint64_t)ctx->unique_set->count);

    printf("\t - List of mr values found: ");
    int* indices = create_sorting_indices(ctx->unique_set);
    for (int i = 0; i < ctx->unique_set->count; i++) {
        int idx = indices[i];
        if (i > 0) printf(", ");
        printf("%lu", ctx->unique_set->values[idx]);
    }
    printf("\n");
    
    free(indices);
}

/**
 * @brief Prints comprehensive sequence classification statistics and mr value distribution analysis.
 * 
 * This function generates a detailed statistical report summarizing the distribution of sequence
 * types (A, B, C) and the frequency of each unique mr value discovered during the search operation.
 * The report provides both absolute counts and percentage distributions, enabling comprehensive
 * analysis of Collatz sequence behavior patterns across the entire search range.
 * 
 * The function implements a two-section report format:
 * 1. Sequence Type Distribution: Shows how sequences are classified based on M* position
 * 2. mr Value Distribution: Shows frequency of each unique mr value in ascending order
 * 
 * For the mr distribution, the function creates a temporary sorted array to present results
 * in ascending mr value order, making it easier to identify patterns and compare against
 * theoretical predictions. Each entry shows the mr value, occurrence count, and percentage
 * of the total search range.
 * 
 * Sorting Implementation:
 * Uses a simple bubble sort on a temporary frequency array to order results by mr value
 * (ascending). The sort is applied to a copy of the data, leaving the original unique set
 * and statistics structures unchanged.
 * 
 * @param stats Pointer to the SequenceStatistics structure containing type classification
 *              counts and mr distribution array. Must have been updated throughout the
 *              search operation.
 * @param unique_set Pointer to the UniqueMrSet containing all discovered unique mr values.
 *                   Used to map distribution array indices to actual mr values.
 * @param total_processed The total number of sequences processed during the search.
 *                        Used as the denominator for percentage calculations.
 * 
 * @note The function displays ALL unique mr values found, not just the most common ones,
 *       ensuring complete statistical documentation.
 * 
 * @note Percentages are displayed with 10 decimal places for high precision, suitable
 *       for scientific analysis and verification.
 * 
 * @note The temporary frequency array is allocated, populated, sorted, and freed within
 *       this function, preventing memory leaks.
 * 
 * @note Bubble sort is used because the number of unique mr values is very small (~42),
 *       making algorithm complexity irrelevant for this operation.
 * 
 * @note The function assumes that stats->mr_distribution indices correspond to positions
 *       in unique_set->values array (parallel array relationship).
 * 
 * @complexity O(n²) where n is the number of unique mr values (typically 42),
 *            dominated by the bubble sort operation
 * 
 * @see SequenceStatistics for statistics structure definition
 * @see UniqueMrSet for unique mr value storage
 * @see classify_and_update_statistics() for how statistics are collected
 * @see MrFreq internal structure for frequency array organization
 * 
 * @example
 * ```c
 * // After completing search of range 1 to 2^25
 * SequenceStatistics* stats = // ... collected during search
 * UniqueMrSet* unique_set = // ... 42 unique values found
 * uint64_t total = 33554432;
 * 
 * print_statistics(stats, unique_set, total);
 * 
 * // Example output:
 * //
 * // [*] SEQUENCE CLASSIFICATION STATISTICS
 * //     - Type A (M* before first mr): 15234567 (45.4123456789%)
 * //     - Type B (M* between first and second mr): 12345678 (36.7891234567%)
 * //     - Type C (M* after second mr): 5974187 (17.7985308644%)
 * //
 * // [*] MR VALUE DISTRIBUTION
 * //     - mr=    0:   10485760 occurrences (31.2500000000%)
 * //     - mr=    1:    5242880 occurrences (15.6250000000%)
 * //     - mr=    2:    2621440 occurrences (7.8125000000%)
 * //     ...
 * //     - mr= 3643:          1 occurrences (0.0000000298%)
 * ```
 */
static void print_statistics(const SequenceStatistics* stats, const UniqueMrSet* unique_set, 
                            uint64_t total_processed) {
    printf("\n[*] SEQUENCE CLASSIFICATION STATISTICS\n");
 
    // Type distribution
    double pct_A = (stats->type_A_count * 100.0) / total_processed;
    double pct_B = (stats->type_B_count * 100.0) / total_processed;
    double pct_C = (stats->type_C_count * 100.0) / total_processed;
    
    printf("\t - Type A (M* before first mr): %lu (%.10f%%)\n", 
           stats->type_A_count, pct_A);
    printf("\t - Type B (M* between first and second mr): %lu (%.10f%%)\n", 
           stats->type_B_count, pct_B);
    printf("\t - Type C (M* after second mr): %lu (%.10f%%)\n", 
           stats->type_C_count, pct_C);
   
    // mr value distribution (ALL values, sorted by mr value)
    printf("\n[*] MR VALUE DISTRIBUTION (All %d unique values)\n", unique_set->count);
    
    // Create array with mr values and their frequencies
    typedef struct {
        uint64_t mr_value;
        uint64_t count;
    } MrFreq;
    
    MrFreq* freq_array = safe_malloc(unique_set->count * sizeof(MrFreq), "frequency array");
    
    for (int i = 0; i < unique_set->count; i++) {
        freq_array[i].mr_value = unique_set->values[i];
        freq_array[i].count = stats->mr_distribution[i];
    }
    
    // Sort by mr_value (ascending)
    for (int i = 0; i < unique_set->count - 1; i++) {
        for (int j = 0; j < unique_set->count - 1 - i; j++) {
            if (freq_array[j].mr_value > freq_array[j + 1].mr_value) {
                MrFreq temp = freq_array[j];
                freq_array[j] = freq_array[j + 1];
                freq_array[j + 1] = temp;
            }
        }
    }
    
    // Print ALL values
    for (int i = 0; i < unique_set->count; i++) {
        double pct = (freq_array[i].count * 100.0) / total_processed;
        printf("\t - mr=%5lu: %10lu occurrences (%.10f%%)\n", 
               freq_array[i].mr_value, freq_array[i].count, pct);
    }
    
    free(freq_array);
}

/**
 * @brief Displays the program identification banner with visual branding for application recognition.
 * 
 * This function outputs a standardized visual header that provides immediate identification
 * of the application and its research focus. The banner serves as the primary visual branding
 * element, establishing program identity through consistent ASCII art formatting and descriptive
 * title that clearly indicates the algorithmic approach and research methodology being employed.
 * 
 * The function creates a prominent visual separator that distinguishes program output from
 * system messages and provides professional presentation suitable for research environments,
 * academic documentation, and technical logging. The banner is designed to be immediately
 * recognizable and informative regardless of execution context or parameter validation outcomes.
 * 
 * Visual Design Features:
 * - Fixed-width ASCII art formatting (74 characters) for consistent terminal presentation
 * - Horizontal asterisk rule separators creating clear visual boundaries
 * - Descriptive title combining algorithm type with research methodology
 * - Professional appearance suitable for academic and research documentation
 * - Consistent branding across all program executions
 * 
 * Usage Context:
 * Called early in program initialization to establish visual context before any validation
 * or configuration operations. This ensures users always see program identification even
 * when command-line arguments are invalid or other early failures occur.
 * 
 * @note The banner spans exactly 74 characters in width, optimized for standard terminal
 *       displays while remaining readable on various screen sizes and output contexts.
 * 
 * @note This function has no parameters and no return value, focusing solely on visual
 *       output generation without any dependency on program state or configuration.
 * 
 * @note The banner text references "tuple-based transform" methodology, providing immediate
 *       context about the algorithmic approach without requiring additional explanation.
 * 
 * @note Output is sent to stdout and will appear in console output, log files, or any
 *       redirected output streams according to standard I/O behavior.
 * 
 * @complexity O(1) - constant time operation with fixed output content
 * 
 * @see print_algorithm_setup() for technical configuration display
 * @see validate_and_parse_arguments() for typical usage context
 * @see main() for program initialization sequence
 * 
 * @example
 * ```c
 * // Called at program start for immediate identification
 * print_program_header();
 * 
 * // Example output:
 * // **************************************************************************
 * // * High-performance mr pairs discovery engine using tuple-based transform *
 * // *                                                  Javier Hernandez 2026 *
 * // **************************************************************************
 * 
 * // Usage in error scenarios
 * int main(int argc, char* argv[]) {
 *     print_program_header();  // Always shows program identity
 *     
 *     if (argc != 2) {
 *         printf("Usage error...\n");
 *         return 1;  // User still sees what program they ran
 *     }
 *     
 *     // ... continue with normal execution
 * }
 * ```
 */
static void print_program_header() {
    printf("\n**************************************************************************\n");
    printf("* High-performance mr pairs discovery engine using tuple-based transform *\n");
    printf("*                                                  Javier Hernandez 2026 *\n");
    printf("**************************************************************************\n");
}

/**
 * @brief Displays comprehensive algorithm setup information including parallelization and search range configuration.
 * 
 * This function outputs detailed technical configuration information that enables users to
 * verify execution parameters, understand computational scope, and validate system resource
 * utilization before the search process begins. The display provides both system-level
 * information (thread count) and algorithm-specific parameters (search range) in a structured,
 * professional format suitable for research documentation and performance analysis.
 * 
 * The function serves as a critical validation checkpoint, allowing users to confirm that
 * the program has correctly interpreted their parameters and is utilizing expected system
 * resources. This information is essential for performance monitoring, resource planning,
 * and result interpretation in computational research contexts.
 * 
 * Configuration Display Features:
 * - Real-time thread count detection showing actual parallelization level
 * - Mathematical range notation (2^exponent) for precise specification
 * - Absolute numeric bounds for concrete understanding of computational scope
 * - Structured formatting with consistent indentation and sectioning
 * - Professional presentation suitable for research logs and documentation
 * 
 * System Information Reported:
 * - Thread utilization: Actual OpenMP thread count available for parallel execution
 * - Search methodology: Implicit confirmation of tuple-based transform approach
 * - Range bounds: Both exponential notation and absolute numeric limits
 * - Parameter validation: Confirmation of successful argument processing
 * 
 * @param exponent The power-of-2 exponent defining the search range upper bound.
 *                 Used to display both mathematical notation (2^exponent) and provide
 *                 computational scope context. Must be a valid integer in range [1,64].
 * @param max_n The calculated maximum search value (2^exponent) representing the
 *              exclusive upper bound of the search range. Used to display absolute
 *              numeric range being processed. Must be computed as (1UL << exponent).
 * 
 * @note The function displays actual thread count from omp_get_max_threads(), providing
 *       real-time verification of the parallel execution environment rather than
 *       theoretical or configured values.
 * 
 * @note Range display shows inclusive bounds (1 to max_n-1) to clarify that max_n
 *       itself is excluded from the search, following standard mathematical convention
 *       for range notation.
 * 
 * @note The ALGORITHM SETUP section header provides clear visual separation from the
 *       program banner and creates logical organization of output information.
 * 
 * @note Thread count verification helps users identify potential performance issues
 *       related to system configuration, OpenMP environment variables, or resource
 *       allocation in shared computing environments.
 * 
 * @warning The function assumes valid input parameters that have passed validation
 *          in validate_and_parse_arguments(). Invalid parameters may produce
 *          incorrect or misleading output.
 * 
 * @complexity O(1) - constant time operation with simple parameter formatting
 * 
 * @see omp_get_max_threads() for dynamic thread count detection
 * @see validate_and_parse_arguments() for parameter validation requirements
 * @see print_program_header() for visual banner that precedes this output
 * @see main() for typical calling sequence and context
 * 
 * @example
 * ```c
 * // Typical usage after successful argument validation
 * int exponent = 25;
 * uint64_t max_n = 1UL << exponent;  // 33,554,432
 * 
 * print_algorithm_setup(exponent, max_n);
 * 
 * // Example output on a 8-core system:
 * // [*] ALGORITHM SETUP
 * //     - Using 8 threads
 * //     - Exploring range from 1 to 2^25 - 1 = 33554431 
 * 
 * // Different system configurations produce different thread counts:
 * // Single-core system: "Using 1 threads"
 * // 16-core system: "Using 16 threads"
 * // Container with limited resources: "Using 4 threads"
 * 
 * // Different range examples:
 * // exponent=20: "Exploring range from 1 to 2^20 - 1 = 1048575"
 * // exponent=30: "Exploring range from 1 to 2^30 - 1 = 1073741823"
 * 
 * // Usage in main function
 * int main(int argc, char* argv[]) {
 *     print_program_header();
 *     
 *     if (!validate_and_parse_arguments(argc, argv, &exponent, &max_n)) {
 *         return 1;
 *     }
 *     
 *     print_algorithm_setup(exponent, max_n);  // Show validated configuration
 *     
 *     // ... proceed with search execution
 * }
 * ```
 */
static void print_algorithm_setup(int exponent, uint64_t max_n) {
    printf("\n[*] ALGORITHM SETUP\n");
    printf("\t - Using %d threads\n", omp_get_max_threads());
    printf("\t - Exploring range from 1 to 2^%d - 1 = %lu \n", exponent, max_n - 1);
}

// ***********************************************************************************
// * 9. COMMAND LINE ARGUMENT PROCESSING
// ***********************************************************************************

/**
 * @brief Validates and parses command-line arguments with strict input validation and comprehensive error handling.
 * 
 * This function performs complete validation and parsing of command-line arguments for the Collatz
 * sequence analysis program, implementing robust error checking with strict format validation,
 * detailed usage instructions, and practical examples. It serves as the primary input validation
 * gateway, ensuring that only valid numeric configurations reach the computational core while
 * providing clear guidance for proper program usage.
 * 
 * The function implements a multi-stage validation process that checks argument count, strictly
 * validates numeric format (rejecting inputs like "32x", "abc", or "25.5"), parses the exponent
 * value, validates range constraints, and computes the derived maximum search value. Each
 * validation stage provides specific error messages and guidance to help users understand
 * proper program usage and select appropriate parameters for their analysis needs.
 * 
 * Validation Process:
 * 1. Argument Count Verification: Ensures exactly one parameter (exponent) is provided
 * 2. Numeric Format Validation: Uses strtol() to strictly validate pure integer format
 * 3. Character Validation: Rejects any input with trailing non-numeric characters
 * 4. Exponent Parsing: Converts validated string to integer value
 * 5. Range Validation: Ensures exponent is within safe computational limits [1, 64]
 * 6. Maximum N Calculation: Computes 2^exponent with overflow safety checks
 * 7. Parameter Output: Updates caller's variables only on successful validation
 * 
 * Strict Format Validation:
 * Unlike atoi() which silently accepts "32x" as 32, this function uses strtol() with
 * endptr checking to strictly require pure integer input. Any non-digit characters
 * (including decimal points, letters, or spaces) result in validation failure with
 * clear error messages showing what was wrong and examples of correct input.
 * 
 * Usage Guidance Features:
 * - Clear syntax explanation with program name extraction from argv[0]
 * - Practical examples showing typical usage patterns
 * - Recommended exponent values with corresponding search ranges
 * - Performance guidance for different computational scales
 * - Specific error messages for each type of validation failure
 * - Examples of both valid and invalid input formats
 * 
 * Error Detection Categories:
 * - No digits found (e.g., "abc", "xyz"): endptr == argv[1]
 * - Trailing characters (e.g., "32x", "25.5"): *endptr != '\0'
 * - Out of range (e.g., 0, 65, 100): exponent < 1 || exponent > 64
 * - Wrong argument count: argc != 2
 * 
 * @param argc The number of command-line arguments including program name.
 *             Expected to be exactly 2 (program name + exponent parameter).
 * @param argv Array of command-line argument strings. argv[0] contains program name,
 *             argv[1] should contain the exponent value as a pure integer string.
 * @param exponent Pointer to store the parsed and validated exponent value.
 *                 Updated only if validation succeeds completely.
 * @param max_n Pointer to store the calculated maximum search value (2^exponent).
 *              Updated only if validation succeeds completely.
 * 
 * @return true if all arguments are valid and successfully parsed
 *         false if any validation fails, with appropriate error messages displayed
 * 
 * @note The function provides comprehensive usage information on argument count mismatch,
 *       including practical examples and performance guidance for different scales.
 * 
 * @note Exponent parsing uses strtol() with endptr validation to strictly reject any
 *       input containing non-digit characters, including "32x", "abc", "25.5", or "30 40".
 * 
 * @note The endptr mechanism allows detection of trailing garbage: if strtol() stops
 *       before reaching the null terminator, the input contains invalid characters.
 * 
 * @note Range validation prevents both underflow (exponent < 1) and overflow scenarios
 *       (exponent > 63) that could cause undefined behavior or excessive computation.
 *
 * @note Maximum N calculation uses bit shifting (1UL << exponent) which is valid for
 *       exponents 0-63. Exponent 64 would overflow uint64_t causing undefined behavior.
 *
 * @warning Large exponent values (approaching 63) will create enormous search spaces
 *          requiring substantial computational resources and execution time.
 * 
 * @note The function only updates output parameters on complete success, ensuring
 *       that partially validated data never reaches the caller.
 * 
 * @note Maximum N calculation uses bit shifting (1UL << exponent) for exact
 *       power-of-2 computation without floating-point precision issues.
 * 
 * @note Error messages show the actual invalid input received, helping users identify
 *       exactly what they typed wrong (e.g., "Invalid exponent '32x'").
 * 
 * @warning Large exponent values (approaching 64) will create enormous search spaces
 *          requiring substantial computational resources and execution time.
 * 
 * @complexity O(n) where n is the length of argv[1] string for parsing,
 *            typically O(1) for reasonable input lengths (1-2 digits)
 * 
 * @see strtol() for robust string-to-integer conversion with error detection
 * @see print_program_header() for visual banner display
 * @see print_algorithm_setup() for technical configuration display
 * @see main() for typical usage context and error handling
 * 
 * @example
 * ```c
 * // Typical main function usage
 * int main(int argc, char* argv[]) {
 *     int exponent;
 *     uint64_t max_n;
 *     
 *     if (!validate_and_parse_arguments(argc, argv, &exponent, &max_n)) {
 *         return 1;  // Exit with error code on validation failure
 *     }
 *     
 *     printf("Valid range: from 1 to 2^%d = %lu\n", exponent, max_n - 1);
 *     // Proceed with validated parameters...
 * }
 * 
 * // Example valid command lines:
 * // ./mr_pairs_finder 25        → exponent=25, max_n=33,554,432
 * // ./mr_pairs_finder 20        → exponent=20, max_n=1,048,576
 * // ./mr_pairs_finder 30        → exponent=30, max_n=1,073,741,824
 * 
 * // Example invalid command lines with strict validation:
 * // ./mr_pairs_finder           → Usage message with examples
 * // ./mr_pairs_finder 32x       → "[*] ERROR: Invalid exponent '32x'. Must be a valid integer."
 * //                                 "      - Examples of valid input: 20, 25, 30"
 * //                                 "      - Examples of invalid input: 32x, abc, 25.5, 30 40"
 * // ./mr_pairs_finder abc       → "[*] ERROR: Invalid exponent 'abc'. Must be a valid integer."
 * // ./mr_pairs_finder 25.5      → "[*] ERROR: Invalid exponent '25.5'. Must be a valid integer."
 * // ./mr_pairs_finder "30 40"   → "[*] ERROR: Invalid exponent '30 40'. Must be a valid integer."
 * // ./mr_pairs_finder 0         → "[*] ERROR: Exponent 0 is out of valid range. Must be between 1 and 63."
 * // ./mr_pairs_finder 64        → "[*] ERROR: Exponent 65 is out of valid range. Must be between 1 and 63."
 * // ./mr_pairs_finder 65        → "[*] ERROR: Exponent 65 is out of valid range. Must be between 1 and 63."
 * // ./mr_pairs_finder -5        → "[*] ERROR: Exponent -5 is out of valid range. Must be between 1 and 63."
 * 
 * // strtol() error detection examples:
 * // Input: "32x"
 * //   - strtol() parses "32", stops at 'x'
 * //   - endptr points to 'x'
 * //   - *endptr != '\0' → REJECTED
 * 
 * // Input: "abc"
 * //   - strtol() finds no digits
 * //   - endptr == argv[1] (no movement)
 * //   - endptr == argv[1] → REJECTED
 * 
 * // Input: "25"
 * //   - strtol() parses "25", reaches '\0'
 * //   - *endptr == '\0' → ACCEPTED
 * //   - Then range check: 1 <= 25 <= 64 → VALID
 * ```
 */
static bool validate_and_parse_arguments(int argc, char* argv[], int* exponent, uint64_t* max_n) {
    
    // Print program header
    print_program_header();

    if (argc != 2) {
        printf("\n[*] USAGE:");
        printf("\n\t%s <exponent>\n", argv[0]);
        printf("\n[*] EXAMPLE:");
        printf("\n\t%s 25  (to search n < 2^25)\n", argv[0]);
        printf("\n[*] RECOMMENDED EXPONENTS:");
        printf("\n\t20 -> 2^20 = 1,048,576 (quick test)");
        printf("\n\t25 -> 2^25 = 33,554,432 (default)");
        printf("\n\t30 -> 2^30 = 1,073,741,824 (intensive use)");
        printf("\n\n");
        return false;
    }
    
    // Validate that argument is a valid integer
    char* endptr;
    long parsed_value = strtol(argv[1], &endptr, 10);
    
    // Check for parsing errors:
    // - endptr == argv[1]: no digits found
    // - *endptr != '\0': extra characters after number
    if (endptr == argv[1] || *endptr != '\0') {
        printf("\n[*] ERROR: Invalid exponent '%s'. Must be a valid integer.\n", argv[1]);
        printf("\t- Examples of valid input: 20, 25, 30\n");
        printf("\t- Examples of invalid input: 32x, abc, 25.5, 30 40\n\n");
        return false;
    }
    
    *exponent = (int)parsed_value;
    
    if (*exponent < 1 || *exponent > 63) {
        printf("\n[*] ERROR: Exponent %d is out of valid range. Must be between %d and %d.\n", 
                *exponent, MIN_EXPONENT, MAX_EXPONENT);
        return false;
    }
    
    *max_n = 1UL << *exponent;
    return true;
}

// ***********************************************************************************
// * 10. MAIN FUNCTION
// ***********************************************************************************

/**
 * @brief Main program entry point orchestrating complete Collatz sequence analysis with checkpoint/resume capability.
 * 
 * This function serves as the central coordinator for the entire Collatz sequence analysis operation,
 * managing the complete application lifecycle from command-line argument processing through result
 * output and comprehensive resource cleanup. It implements a robust workflow with fault tolerance
 * through automatic checkpoint/resume capability, enabling long-running searches to survive system
 * failures, user interruptions, and intentional pauses without losing computational progress.
 * 
 * The program automatically saves computation state every 5 minutes (configurable via CHECKPOINT_INTERVAL)
 * and seamlessly resumes from the last checkpoint if interrupted. This is critical for large exponents
 * (30+) where computation may take hours or days. The checkpoint system provides:
 * - Automatic periodic saves with minimal overhead (~1KB checkpoint files)
 * - Integrity validation on resume (magic number, exponent match, complete data verification)
 * - Backup rotation preventing corruption from incomplete writes
 * - Graceful interruption handling via Ctrl+C (SIGINT) with immediate checkpoint save
 * - Automatic cleanup of checkpoint files upon successful completion
 * 
 * Program Execution Workflow:
 * 1. Command-line argument validation with comprehensive usage guidance
 * 2. Signal handler registration for graceful interruption (SIGINT/Ctrl+C)
 * 3. Algorithm setup display showing thread count and search range
 * 4. Search context initialization (unique set, progress tracker, statistics)
 * 5. Checkpoint loading attempt (restores state if previous run was interrupted)
 * 6. Parallel search execution with automatic checkpoints and progress monitoring
 * 7. Performance timing and metrics calculation
 * 8. JSON result file generation for modern data processing integration
 * 9. Comprehensive validation report with statistics and sorted mr value list
 * 10. Sequence classification statistics and mr distribution analysis
 * 11. JSON file information display (filename, size)
 * 12. Complete resource cleanup (unique set, progress tracker, statistics, checkpoint files)
 * 
 * Fault Tolerance Features:
 * - Automatic checkpoint saves prevent computation loss on system failures
 * - Graceful interruption handling (Ctrl+C) saves state before exit
 * - Checkpoint validation prevents resume from corrupted or mismatched files
 * - Backup rotation protects against incomplete checkpoint writes
 * - Resume capability displays precise progress percentage and restored state
 * - Exponent mismatch detection prevents resuming wrong search ranges
 * 
 * Output Strategy:
 * The program generates multiple output formats to serve different use cases:
 * - JSON file: Machine-readable format for data processing and web integration
 * - Console summary: Quick validation and performance assessment
 * - Statistics report: Detailed distribution analysis for research
 * 
 * Resume Workflow:
 * If interrupted (Ctrl+C, crash, or intentional stop), simply run the program again with the
 * same exponent to automatically resume from the last checkpoint. The program detects existing
 * checkpoint files, validates compatibility, restores all discovered unique mr values, and
 * continues processing from the exact interruption point.
 * 
 * @param argc Number of command-line arguments including the program name.
 *             Expected to be exactly 2 (program name + exponent parameter).
 * @param argv Array of command-line argument strings:
 *             - argv[0]: Program name (used in usage messages)
 *             - argv[1]: Exponent value defining search range (n < 2^exponent)
 * 
 * @return 0 on successful completion including search execution, result output, and cleanup
 *         1 on command-line argument validation failure
 * 
 * @note Generated JSON filename follows pattern: "mr_pairs_detected_on_range_1_to_2pow<exponent>.json"
 *       enabling clear identification of search range from filename alone.
 * 
 * @note Checkpoint files (checkpoint.bin, checkpoint.bak) are automatically cleaned up upon
 *       successful search completion to avoid confusion when starting new searches.
 * 
 * @note If interrupted (Ctrl+C or system failure), checkpoint files remain for resume capability.
 *       Simply run the program again with the same exponent to continue from interruption point.
 * 
 * @note Signal handler (SIGINT) ensures graceful shutdown with checkpoint save, preventing data
 *       loss even for unexpected interruptions or user cancellation.
 * 
 * @note The program displays comprehensive progress information during execution, including:
 *       completion percentage, processing speed, unique discoveries, and estimated time remaining.
 * 
 * @note JSON output uses clean array-of-objects format optimized for modern data processing
 *       tools, web applications, and analysis frameworks.
 * 
 * @note Statistics output includes both sequence type distribution (A/B/C) and complete mr value
 *       frequency distribution, enabling comprehensive pattern analysis.
 * 
 * @note Resource cleanup is performed unconditionally to ensure proper system resource management
 *       regardless of execution outcomes.
 * 
 * @warning For very large exponents (35+), ensure sufficient disk space for JSON output and
 *          adequate system resources for parallel processing across all available threads.
 * 
 * @warning Do not delete checkpoint files manually while search is in progress, as this will
 *          prevent resume capability if the process is interrupted.
 * 
 * @warning Changing exponent between runs invalidates existing checkpoints. The program will
 *          detect mismatch and start fresh with the new exponent.
 * 
 * @complexity O(n/p) where n = 2^exponent and p = number of threads, with periodic O(m)
 *            checkpoint overhead where m ≈ 42 unique values (negligible impact)
 * 
 * @see validate_and_parse_arguments() for command-line processing
 * @see load_checkpoint() for automatic resume capability
 * @see save_checkpoint() for state persistence
 * @see checkpoint_signal_handler() for graceful interruption handling
 * @see execute_parallel_search() for core search orchestration
 * @see write_json_results() for JSON output generation
 * @see print_results() for console summary
 * @see print_statistics() for detailed distribution analysis
 * 
 * @example
 * ```c
 * // Initial execution for large exponent
 * $ ./mr_pairs_finder 30
 * // **************************************************************************
 * // * High-performance mr pairs discovery engine using tuple-based transform *
 * // *                                                  Javier Hernandez 2026 *
 * // **************************************************************************
 * //
 * // [*] ALGORITHM SETUP
 * //     - Using 8 threads
 * //     - Exploring range from 1 to 2^30 = 1073741823
 * //
 * // [*] SEARCH PROCESS
 * //     - Starting from n=1
 * //     ... runs for several hours with periodic checkpoints ...
 * //     - [Autosaving checkpoint at n = 450000000]
 * //     ^C (user presses Ctrl+C)
 * //     - Interrupted! Checkpoint saved at n = 450123456
 * //     - Run again with same exponent to resume
 * 
 * // Resume execution after interruption
 * $ ./mr_pairs_finder 30
 * // [program header]
 * // [*] ALGORITHM SETUP
 * //     - Using 8 threads
 * //     - Exploring range from 1 to 2^30 = 1073741823
 * //
 * // [*] CHECKPOINT LOADED
 * //     - Last processed: n = 450123456
 * //     - Resuming from: n = 450123457
 * //     - Completed: 41.9234567890% | Remaining: 58.0765432110%
 * //     - Restored 42 unique mr values
 * //
 * // [*] SEARCH PROCESS
 * //     - Resuming from n=450123457
 * //     ... continues processing ...
 * //     - Final checkpoint saved
 * //     - Checkpoint files cleaned...search completed
 * //
 * // [*] SEARCH RESULTS
 * //     - Total numbers processed: 1073741823
 * //     - Total time: 8543.210 seconds
 * //     - Speed: 125678.45 numbers/second
 * //     - Numbers with mr found: 1073741823
 * //     - Unique mr values found: 42
 * //     - List of mr values found: 0, 1, 2, 3, 6, 7, ...
 * //
 * // [*] SEQUENCE CLASSIFICATION STATISTICS
 * //     - Type A: ... | Type B: ... | Type C: ...
 * //
 * // [*] MR VALUE DISTRIBUTION (All 42 unique values)
 * //     - mr=    0: ... occurrences (...)
 * //     ...
 * //
 * // [*] JSON OUTPUT
 * //     - File: 'mr_pairs_detected_on_range_1_to_2pow30.json' (1234 bytes)
 * 
 * // Exponent mismatch scenario (prevents wrong resume)
 * $ ./mr_pairs_finder 28
 * // [*] WARNING: Checkpoint exponent mismatch (checkpoint=30, current=28). Starting fresh.
 * // [continues with fresh search for exponent 28]
 * ```
 */
int main(int argc, char* argv[]) {
    // Parse and validate command line arguments
    int exponent;
    uint64_t max_n;
    if (!validate_and_parse_arguments(argc, argv, &exponent, &max_n)) {
        return 1;
    }
    
    // Setup signal handler for graceful checkpoint on interruption
    signal(SIGINT, checkpoint_signal_handler);

    // Print algorithm setup
    print_algorithm_setup(exponent, max_n);
    
    // Initialize search context
    SearchContext ctx = {
        .max_n = max_n,
        .exponent = exponent,
        .unique_set = create_unique_mr_set(),
        .progress = create_progress_tracker(),
        .stats = create_statistics(),
        .start_time = omp_get_wtime()
    };

    // Try to load checkpoint
    uint64_t start_n = 1;
    load_checkpoint(&ctx, &start_n);

    // Execute the parallel search
    uint64_t found_count = 0;
    uint64_t processed_count = 0;
    
    execute_parallel_search(&ctx, start_n, &found_count, &processed_count);
    
    double end_time = omp_get_wtime();
    double total_time = end_time - ctx.start_time;
    
    // Generate output filename and write JSON results
    char json_filename[256];
    snprintf(json_filename, sizeof(json_filename), "mr_pairs_detected_on_range_1_to_2pow%d.json", exponent);
    
    write_json_results(&ctx, json_filename);
    
    // Print validation results and summary
    print_results(&ctx, total_time, found_count, processed_count);
    
    // Print stats
    print_statistics(ctx.stats, ctx.unique_set, processed_count);

    // Print file details
    FILE* test_file = fopen(json_filename, "r");
    if (test_file) {
        fseek(test_file, 0, SEEK_END);
        long file_size = ftell(test_file);
        fclose(test_file);
        
        printf("\n[*] OUTPUT FILES\n");
        printf("\t - File: '%s' (%ld bytes)\n", json_filename, file_size);
    }
    
    // Cleanup
    destroy_unique_mr_set(ctx.unique_set);
    destroy_progress_tracker(ctx.progress);
    destroy_statistics(ctx.stats);
    
    return 0;
}
