/**
 * Web Worker for parallel Langlands correspondence computations
 * Enables offloading heavy mathematical operations from the main thread
 */

// Import WASM module in worker context
let wasmModule = null;
let engine = null;

// Message handler for communication with main thread
self.addEventListener('message', async function(event) {
    const { type, data, id } = event.data;
    
    try {
        switch (type) {
            case 'init':
                await initializeWorker();
                postMessage({ type: 'init', success: true, id });
                break;
                
            case 'compute':
                const result = await computeInWorker(data);
                postMessage({ type: 'compute', result, id });
                break;
                
            case 'batch_compute':
                const batchResults = await batchComputeInWorker(data);
                postMessage({ type: 'batch_compute', results: batchResults, id });
                break;
                
            case 'math_operation':
                const mathResult = await performMathOperation(data);
                postMessage({ type: 'math_operation', result: mathResult, id });
                break;
                
            case 'terminate':
                self.close();
                break;
                
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        postMessage({ 
            type: 'error', 
            error: error.message, 
            stack: error.stack,
            id 
        });
    }
});

/**
 * Initialize the WASM module in worker context
 */
async function initializeWorker() {
    if (wasmModule) return; // Already initialized
    
    // Import WASM module (adjust path as needed)
    const wasmImport = await import('../pkg/geometric_langlands_wasm.js');
    
    // Initialize WASM
    wasmModule = await wasmImport.default();
    
    // Create engine instance
    const { LanglandsEngine, WasmConfig } = wasmImport;
    
    const config = new WasmConfig();
    config.set_max_workers(1); // Single worker mode
    config.set_memory_limit(200); // Limit memory per worker
    config.set_enable_gpu(false); // Disable GPU in worker
    
    engine = new LanglandsEngine(config);
    await engine.initialize();
    
    console.log('🤖 WASM Worker initialized successfully');
}

/**
 * Perform Langlands correspondence computation in worker
 */
async function computeInWorker({ groupType, dimension, options = {} }) {
    if (!engine) {
        throw new Error('Worker not initialized');
    }
    
    // Progress reporting
    const progressCallback = (progress) => {
        postMessage({ 
            type: 'progress', 
            progress: progress,
            step: `Computing ${groupType}(${dimension}) correspondence...`
        });
    };
    
    // Simulate progress reporting during computation
    const steps = [
        'Setting up geometric structures...',
        'Computing automorphic representations...',
        'Analyzing Galois actions...',
        'Verifying correspondence...',
        'Finalizing results...'
    ];
    
    for (let i = 0; i < steps.length; i++) {
        postMessage({ 
            type: 'progress', 
            progress: (i + 1) / steps.length,
            step: steps[i]
        });
        
        // Small delay to demonstrate progress
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    const result = await engine.compute_correspondence(groupType, dimension);
    
    // Add worker-specific metadata
    result.worker_computed = true;
    result.worker_id = self.name || 'worker';
    result.memory_usage = engine.estimate_memory_usage();
    
    return result;
}

/**
 * Perform batch computations for multiple correspondences
 */
async function batchComputeInWorker({ computations }) {
    if (!engine) {
        throw new Error('Worker not initialized');
    }
    
    const results = [];
    const total = computations.length;
    
    for (let i = 0; i < computations.length; i++) {
        const { groupType, dimension } = computations[i];
        
        postMessage({ 
            type: 'batch_progress', 
            progress: i / total,
            current: i + 1,
            total: total,
            step: `Computing ${groupType}(${dimension})...`
        });
        
        try {
            const result = await engine.compute_correspondence(groupType, dimension);
            result.batch_index = i;
            results.push(result);
        } catch (error) {
            results.push({
                batch_index: i,
                error: error.message,
                groupType,
                dimension
            });
        }
    }
    
    return results;
}

/**
 * Perform mathematical operations using WASM utilities
 */
async function performMathOperation({ operation, params }) {
    // Import math utilities
    const { MathUtils } = await import('../pkg/geometric_langlands_wasm.js');
    
    switch (operation) {
        case 'primes_up_to':
            return {
                operation: 'primes_up_to',
                input: params.n,
                result: MathUtils.primes_up_to(params.n),
                computation_time: performance.now()
            };
            
        case 'euler_totient':
            return {
                operation: 'euler_totient',
                input: params.n,
                result: MathUtils.euler_totient(params.n),
                computation_time: performance.now()
            };
            
        case 'is_prime':
            return {
                operation: 'is_prime',
                input: params.n,
                result: MathUtils.is_prime(params.n),
                computation_time: performance.now()
            };
            
        case 'batch_totient':
            const totients = [];
            const startTime = performance.now();
            
            for (let i = params.start; i <= params.end; i++) {
                totients.push({
                    n: i,
                    phi: MathUtils.euler_totient(i)
                });
                
                // Report progress for large batches
                if (i % 100 === 0) {
                    postMessage({ 
                        type: 'math_progress', 
                        progress: (i - params.start) / (params.end - params.start),
                        current: i
                    });
                }
            }
            
            return {
                operation: 'batch_totient',
                range: [params.start, params.end],
                result: totients,
                computation_time: performance.now() - startTime
            };
            
        default:
            throw new Error(`Unknown math operation: ${operation}`);
    }
}

/**
 * Helper function to simulate heavy computation with progress reporting
 */
async function simulateHeavyComputation(steps, callback) {
    for (let i = 0; i < steps; i++) {
        // Simulate work
        await new Promise(resolve => {
            let sum = 0;
            for (let j = 0; j < 1000000; j++) {
                sum += Math.random();
            }
            setTimeout(resolve, 10);
        });
        
        if (callback) {
            callback((i + 1) / steps);
        }
    }
}

/**
 * Performance monitoring for worker operations
 */
class WorkerPerformanceMonitor {
    constructor() {
        this.metrics = new Map();
        this.startTime = performance.now();
    }
    
    mark(label) {
        const elapsed = performance.now() - this.startTime;
        this.metrics.set(label, elapsed);
        
        postMessage({
            type: 'performance_mark',
            label,
            elapsed,
            memory: this.getMemoryUsage()
        });
    }
    
    getMemoryUsage() {
        // Rough estimate of memory usage in worker
        if (engine) {
            return engine.estimate_memory_usage();
        }
        return 0;
    }
    
    getMetrics() {
        return Object.fromEntries(this.metrics);
    }
}

// Global performance monitor
const performanceMonitor = new WorkerPerformanceMonitor();

// Error handler for uncaught errors
self.addEventListener('error', function(event) {
    postMessage({ 
        type: 'worker_error', 
        error: event.error.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
    });
});

// Unhandled promise rejection handler
self.addEventListener('unhandledrejection', function(event) {
    postMessage({ 
        type: 'worker_promise_error', 
        error: event.reason.toString()
    });
    event.preventDefault();
});

// Signal that worker is ready
postMessage({ type: 'worker_ready' });