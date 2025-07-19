/**
 * JavaScript API for Geometric Langlands WASM computations
 * Provides high-level interface with automatic initialization and error handling
 */

let wasmModule = null;
let initPromise = null;

/**
 * Initialize the WASM module (singleton pattern)
 */
export async function initializeWASM() {
    if (wasmModule) return wasmModule;
    if (initPromise) return initPromise;
    
    initPromise = (async () => {
        const init = await import('../pkg/geometric_langlands_conjecture.js');
        await init.default();
        wasmModule = await import('../pkg/geometric_langlands_conjecture.js');
        return wasmModule;
    })();
    
    return initPromise;
}

/**
 * High-level API for Geometric Langlands computations
 */
export class GeometricLanglandsAPI {
    constructor() {
        this.computer = null;
        this.initialized = false;
        this.gpuAvailable = false;
        this.performanceMonitor = new PerformanceMonitor();
    }
    
    /**
     * Initialize the API and optionally GPU acceleration
     */
    async initialize(config = {}) {
        const wasm = await initializeWASM();
        this.computer = new wasm.GeometricLanglandsComputer();
        
        // Try to initialize GPU
        if (config.useGPU !== false) {
            try {
                await this.computer.init_gpu();
                this.gpuAvailable = true;
                console.log('GPU acceleration enabled');
            } catch (e) {
                console.warn('GPU not available, using CPU:', e);
                this.gpuAvailable = false;
            }
        }
        
        // Apply configuration
        if (config.precision || config.maxIterations) {
            await this.setConfig(config);
        }
        
        this.initialized = true;
        return this;
    }
    
    /**
     * Set computation configuration
     */
    async setConfig(config) {
        if (!this.initialized) {
            throw new Error('API not initialized. Call initialize() first.');
        }
        
        const currentConfig = JSON.parse(this.computer.get_config());
        const newConfig = { ...currentConfig, ...config };
        this.computer.set_config(JSON.stringify(newConfig));
    }
    
    /**
     * Compute moduli space properties
     */
    async computeModuliSpace(params) {
        return this.performanceMonitor.measure('moduli_space', async () => {
            const wasm = await initializeWASM();
            const { genus, rank, degree } = params;
            
            const dimension = wasm.compute_moduli_dimension(genus, rank, degree);
            
            // Additional computations can be added here
            return {
                dimension,
                genus,
                rank,
                degree,
                stable: dimension > 0,
                computationTime: this.performanceMonitor.getLastMeasurement()
            };
        });
    }
    
    /**
     * Compute sheaf cohomology groups
     */
    async computeSheafCohomology(params) {
        return this.performanceMonitor.measure('sheaf_cohomology', async () => {
            const wasm = await initializeWASM();
            const { dimension, degree } = params;
            
            const cohomology = wasm.compute_sheaf_cohomology(
                dimension, 
                degree, 
                this.gpuAvailable
            );
            
            return {
                groups: cohomology,
                dimension,
                degree,
                useGPU: this.gpuAvailable,
                computationTime: this.performanceMonitor.getLastMeasurement()
            };
        });
    }
    
    /**
     * Compute Hitchin fibration map
     */
    async computeHitchinMap(bundleData) {
        return this.performanceMonitor.measure('hitchin_map', async () => {
            const wasm = await initializeWASM();
            
            const result = await wasm.compute_hitchin_map(
                JSON.stringify(bundleData),
                this.gpuAvailable
            );
            
            return {
                ...JSON.parse(result),
                useGPU: this.gpuAvailable,
                computationTime: this.performanceMonitor.getLastMeasurement()
            };
        });
    }
    
    /**
     * Run computation in web worker for non-blocking execution
     */
    async computeInWorker(workerScript, taskData) {
        const wasm = await initializeWASM();
        const worker = new wasm.ComputeWorker(workerScript);
        
        return new Promise((resolve, reject) => {
            worker.on_message((result) => {
                try {
                    resolve(JSON.parse(result));
                } catch (e) {
                    resolve(result);
                }
            });
            
            worker.compute(JSON.stringify(taskData));
        });
    }
    
    /**
     * Get current performance metrics
     */
    getMetrics() {
        if (!this.initialized) {
            throw new Error('API not initialized');
        }
        
        const wasmMetrics = JSON.parse(this.computer.get_metrics());
        const performanceMetrics = this.performanceMonitor.getSummary();
        
        return {
            ...wasmMetrics,
            performance: performanceMetrics,
            gpuAvailable: this.gpuAvailable
        };
    }
    
    /**
     * Batch compute multiple operations
     */
    async batchCompute(operations) {
        const results = [];
        const startTime = performance.now();
        
        for (const op of operations) {
            switch (op.type) {
                case 'moduli':
                    results.push(await this.computeModuliSpace(op.params));
                    break;
                case 'cohomology':
                    results.push(await this.computeSheafCohomology(op.params));
                    break;
                case 'hitchin':
                    results.push(await this.computeHitchinMap(op.params));
                    break;
                default:
                    throw new Error(`Unknown operation type: ${op.type}`);
            }
        }
        
        return {
            results,
            totalTime: performance.now() - startTime,
            operations: operations.length
        };
    }
}

/**
 * Performance monitoring utility
 */
class PerformanceMonitor {
    constructor() {
        this.measurements = new Map();
        this.lastMeasurement = 0;
    }
    
    async measure(name, fn) {
        const start = performance.now();
        const result = await fn();
        const duration = performance.now() - start;
        
        this.lastMeasurement = duration;
        
        if (!this.measurements.has(name)) {
            this.measurements.set(name, []);
        }
        this.measurements.get(name).push(duration);
        
        return result;
    }
    
    getLastMeasurement() {
        return this.lastMeasurement;
    }
    
    getSummary() {
        const summary = {};
        
        for (const [name, times] of this.measurements) {
            const sorted = times.sort((a, b) => a - b);
            summary[name] = {
                count: times.length,
                total: times.reduce((a, b) => a + b, 0),
                average: times.reduce((a, b) => a + b, 0) / times.length,
                min: sorted[0],
                max: sorted[sorted.length - 1],
                median: sorted[Math.floor(sorted.length / 2)]
            };
        }
        
        return summary;
    }
}

/**
 * Utility functions
 */
export const utils = {
    /**
     * Format large numbers for display
     */
    formatNumber(num) {
        if (num > 1e9) return (num / 1e9).toFixed(2) + 'B';
        if (num > 1e6) return (num / 1e6).toFixed(2) + 'M';
        if (num > 1e3) return (num / 1e3).toFixed(2) + 'K';
        return num.toString();
    },
    
    /**
     * Check WebGPU availability
     */
    async checkWebGPU() {
        if (!navigator.gpu) {
            return { available: false, reason: 'WebGPU not supported' };
        }
        
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                return { available: false, reason: 'No GPU adapter found' };
            }
            
            const device = await adapter.requestDevice();
            return { 
                available: true, 
                adapter: adapter.name,
                features: [...adapter.features]
            };
        } catch (e) {
            return { available: false, reason: e.message };
        }
    },
    
    /**
     * Memory usage estimation
     */
    getMemoryUsage() {
        if (performance.memory) {
            return {
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                limit: performance.memory.jsHeapSizeLimit
            };
        }
        return null;
    }
};

// Export singleton instance for convenience
export const langlandsAPI = new GeometricLanglandsAPI();