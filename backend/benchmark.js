
const { performance } = require("node:perf_hooks");
const { getEmbedding } = require("./embedding");
const { initializeCache, searchCache } = require("./cache");

async function measure(label, fn, iterations = 10) {
    const times = [];

    await fn(); // Warm-up

    for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await fn();
        times.push(performance.now() - start);
    }

    times.sort((a, b) => a - b);

    const average = times.reduce((a, b) => a + b, 0) / times.length;
    const median = times[Math.floor(times.length / 2)];

    process.stdout.write(
        `\n${label}\nAverage: ${average.toFixed(2)} ms\nMedian: ${median.toFixed(2)} ms\n`
    );
}

async function runBenchmark() {
    console.log("========== SMARTLY RECALL BENCHMARK ==========");

    await initializeCache();

    const question = "Where can I track my package?";

    await measure("Embedding Generation", () =>
        getEmbedding(question)
    );

    // Suppress only the FAQ comparison logs
    const originalLog = console.log;
    console.log = () => {};

    try {
        await measure("Semantic Cache Search", () =>
            searchCache(question)
        );
    } finally {
        console.log = originalLog;
    }

    console.log("\n========== BENCHMARK COMPLETE ==========");
}

runBenchmark().catch(console.error);
