
const { performance } = require("node:perf_hooks");

const URL = "http://localhost:5000/api/ask";

async function ask(question) {
    const start = performance.now();

    const response = await fetch(URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question })
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
    }

    return {
        latency: performance.now() - start,
        data
    };
}

async function measure(label, question, expectedType, iterations = 10) {
    const times = [];

    for (let i = 0; i < iterations; i++) {
        const result = await ask(question);

        if (result.data.cacheType !== expectedType) {
            throw new Error(
                `Expected ${expectedType}, got ${result.data.cacheType}`
            );
        }

        times.push(result.latency);
    }

    times.sort((a, b) => a - b);

    const average =
        times.reduce((sum, time) => sum + time, 0) / times.length;

    const middle = Math.floor(times.length / 2);

    const median = times.length % 2 === 0
        ? (times[middle - 1] + times[middle]) / 2
        : times[middle];

    console.log(`\n${label}`);
    console.log("Requests:", iterations);
    console.log("Average:", average.toFixed(2), "ms");
    console.log("Median:", median.toFixed(2), "ms");
}

async function run() {
    console.log("========== API BENCHMARK ==========");

    // 1. Semantic cache benchmark
    const semanticQuestion =
        "Where can I find my order tracking details?";

    const semanticWarmup = await ask(semanticQuestion);

    console.log("\nSemantic warmup:", semanticWarmup.data.cacheType);

    await measure(
        "SEMANTIC CACHE HIT",
        semanticQuestion,
        "semantic"
    );

    // 2. Groq fallback (one request)
    const uniqueQuestion =
        "Can you explain ancient Roman architecture? Benchmark " + Date.now();

    const miss = await ask(uniqueQuestion);

    console.log("\nGROQ FALLBACK");
    console.log("Latency:", miss.latency.toFixed(2), "ms");
    console.log("Cache type:", miss.data.cacheType);

    if (miss.data.cacheType !== "miss") {
        throw new Error("Expected a cache miss for the unique question.");
    }

    // 3. Exact cache benchmark
    await measure(
        "EXACT CACHE HIT",
        uniqueQuestion,
        "exact"
    );

    console.log("\n========== BENCHMARK COMPLETE ==========");
}

run().catch(error => {
    console.error("Benchmark failed:", error.message);
});
