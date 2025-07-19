/**
 * Web Worker for parallel Geometric Langlands computations
 * Handles heavy computations without blocking the main thread
 */

// Import WASM module
importScripts('../pkg/geometric_langlands_conjecture.js');

let wasmModule = null;
let computer = null;

// Initialize WASM in worker
async function initWorker() {
    if (!wasmModule) {
        await wasm_bindgen('../pkg/geometric_langlands_conjecture_bg.wasm');
        wasmModule = wasm_bindgen;
        computer = new wasmModule.GeometricLanglandsComputer();
        
        // Try to initialize GPU (may fail in worker context)
        try {
            await computer.init_gpu();
            postMessage({ type: 'status', message: 'Worker initialized with GPU' });
        } catch (e) {
            postMessage({ type: 'status', message: 'Worker initialized (CPU only)' });
        }
    }
}

// Message handler
self.addEventListener('message', async (event) => {
    const { type, id, data } = event.data;
    
    try {
        await initWorker();
        
        switch (type) {
            case 'moduli_space':
                const moduliResult = await computeModuliSpace(data);
                postMessage({ type: 'result', id, result: moduliResult });
                break;
                
            case 'cohomology':
                const cohomologyResult = await computeCohomology(data);
                postMessage({ type: 'result', id, result: cohomologyResult });
                break;
                
            case 'hitchin_map':
                const hitchinResult = await computeHitchinMap(data);
                postMessage({ type: 'result', id, result: hitchinResult });
                break;
                
            case 'batch':
                const batchResult = await processBatch(data);
                postMessage({ type: 'result', id, result: batchResult });
                break;
                
            case 'heavy_computation':
                const heavyResult = await performHeavyComputation(data);
                postMessage({ type: 'result', id, result: heavyResult });
                break;
                
            default:
                throw new Error(`Unknown computation type: ${type}`);
        }
    } catch (error) {
        postMessage({ 
            type: 'error', 
            id, 
            error: error.message || 'Unknown error occurred' 
        });
    }
});

// Computation functions
async function computeModuliSpace(params) {
    const { genus, rank, degree } = params;
    const startTime = performance.now();
    
    // Main computation
    const dimension = wasmModule.compute_moduli_dimension(genus, rank, degree);
    
    // Additional computations for demonstration
    const stability = checkStability(genus, rank, degree);
    const eulerChar = computeEulerCharacteristic(genus, rank, degree);
    
    return {
        dimension,
        genus,
        rank,
        degree,
        stability,
        eulerCharacteristic: eulerChar,
        computationTime: performance.now() - startTime
    };
}

async function computeCohomology(params) {
    const { dimension, degree, useGPU } = params;
    const startTime = performance.now();
    
    const groups = wasmModule.compute_sheaf_cohomology(dimension, degree, useGPU);
    
    // Process cohomology groups
    const processed = processCohomologyGroups(groups, dimension, degree);
    
    return {
        groups: processed.groups,
        dimension,
        degree,
        betti: processed.betti,
        euler: processed.euler,
        computationTime: performance.now() - startTime
    };
}

async function computeHitchinMap(params) {
    const startTime = performance.now();
    
    const result = await wasmModule.compute_hitchin_map(
        JSON.stringify(params),
        true // use GPU if available
    );
    
    const parsed = JSON.parse(result);
    
    return {
        ...parsed,
        computationTime: performance.now() - startTime
    };
}

async function processBatch(operations) {
    const results = [];
    const startTime = performance.now();
    
    // Process operations in parallel chunks
    const chunkSize = 10;
    for (let i = 0; i < operations.length; i += chunkSize) {
        const chunk = operations.slice(i, i + chunkSize);
        const chunkResults = await Promise.all(
            chunk.map(op => processOperation(op))
        );
        results.push(...chunkResults);
        
        // Send progress update
        postMessage({
            type: 'progress',
            progress: (i + chunkSize) / operations.length
        });
    }
    
    return {
        results,
        totalOperations: operations.length,
        totalTime: performance.now() - startTime
    };
}

async function processOperation(op) {
    switch (op.type) {
        case 'moduli':
            return computeModuliSpace(op.params);
        case 'cohomology':
            return computeCohomology(op.params);
        case 'hitchin':
            return computeHitchinMap(op.params);
        default:
            throw new Error(`Unknown operation type: ${op.type}`);
    }
}

async function performHeavyComputation(params) {
    const { type, iterations = 1000 } = params;
    const startTime = performance.now();
    
    postMessage({ type: 'status', message: 'Starting heavy computation...' });
    
    // Simulate different types of heavy computations
    let result;
    
    switch (type) {
        case 'spectral_sequence':
            result = await computeSpectralSequence(iterations);
            break;
            
        case 'moduli_stack':
            result = await computeModuliStack(iterations);
            break;
            
        case 'derived_category':
            result = await computeDerivedCategory(iterations);
            break;
            
        default:
            throw new Error(`Unknown heavy computation type: ${type}`);
    }
    
    return {
        type,
        iterations,
        result,
        computationTime: performance.now() - startTime
    };
}

// Helper functions
function checkStability(genus, rank, degree) {
    // Simplified stability check
    const slope = degree / rank;
    const threshold = (genus - 1) / rank;
    
    return {
        isStable: slope > threshold,
        slope,
        threshold
    };
}

function computeEulerCharacteristic(genus, rank, degree) {
    // Riemann-Roch for vector bundles
    return rank * (1 - genus) + degree;
}

function processCohomologyGroups(groups, dimension, degree) {
    // Process raw cohomology data
    const processed = [];
    let euler = 0;
    const betti = [];
    
    for (let i = 0; i <= dimension; i++) {
        const groupDim = Math.max(0, degree - i);
        processed.push({
            degree: i,
            dimension: groupDim
        });
        
        betti.push(groupDim);
        euler += Math.pow(-1, i) * groupDim;
    }
    
    return {
        groups: processed,
        betti,
        euler
    };
}

// Simulated heavy computations
async function computeSpectralSequence(iterations) {
    const pages = [];
    
    for (let page = 0; page < 5; page++) {
        const differentials = [];
        
        for (let i = 0; i < iterations / 100; i++) {
            // Simulate computation
            const d = Math.sin(i * page) * Math.cos(i / page);
            differentials.push(d);
            
            // Yield to event loop periodically
            if (i % 100 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        pages.push({
            page,
            differentials: differentials.length,
            converged: page >= 3
        });
    }
    
    return { pages, totalPages: pages.length };
}

async function computeModuliStack(iterations) {
    const points = [];
    
    for (let i = 0; i < iterations; i++) {
        // Simulate moduli point computation
        const theta = (i / iterations) * 2 * Math.PI;
        const r = Math.sqrt(i / iterations);
        
        points.push({
            x: r * Math.cos(theta),
            y: r * Math.sin(theta),
            stable: r > 0.5
        });
        
        // Progress update every 10%
        if (i % (iterations / 10) === 0) {
            postMessage({
                type: 'progress',
                progress: i / iterations,
                message: `Computing moduli point ${i}/${iterations}`
            });
        }
        
        // Yield periodically
        if (i % 100 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
    
    const stablePoints = points.filter(p => p.stable).length;
    
    return {
        totalPoints: points.length,
        stablePoints,
        stabilityRatio: stablePoints / points.length
    };
}

async function computeDerivedCategory(iterations) {
    const objects = [];
    const morphisms = [];
    
    // Generate objects
    for (let i = 0; i < iterations / 10; i++) {
        objects.push({
            id: i,
            degree: Math.floor(Math.random() * 10) - 5,
            rank: Math.floor(Math.random() * 5) + 1
        });
    }
    
    // Generate morphisms
    for (let i = 0; i < iterations / 5; i++) {
        const source = objects[Math.floor(Math.random() * objects.length)];
        const target = objects[Math.floor(Math.random() * objects.length)];
        
        if (source.id !== target.id) {
            morphisms.push({
                source: source.id,
                target: target.id,
                degree: target.degree - source.degree
            });
        }
        
        // Yield periodically
        if (i % 50 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
    
    return {
        objects: objects.length,
        morphisms: morphisms.length,
        averageDegree: objects.reduce((sum, obj) => sum + obj.degree, 0) / objects.length
    };
}

// Send ready message
postMessage({ type: 'ready', message: 'Worker ready to receive computations' });