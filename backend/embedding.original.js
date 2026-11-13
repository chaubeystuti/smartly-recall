
let extractor;

async function getExtractor() {
    if (!extractor) {
        const { pipeline } = await import("@huggingface/transformers");

        console.log("Loading multilingual embedding model...");

        extractor = await pipeline(
            "feature-extraction",
            "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
            { dtype: "q8" }
        );

        console.log("Embedding model loaded!");
    }

    return extractor;
}

async function getEmbedding(text) {
    const model = await getExtractor();

    const output = await model(text, {
        pooling: "mean",
        normalize: true
    });

    return Array.from(output.data);
}

module.exports = { getEmbedding };

// Test the embedding function
if (require.main === module) {
    (async () => {
        try {
            const vector = await getEmbedding("Where is my order?");

            console.log("Embedding generated!");
            console.log("Vector dimensions:", vector.length);
            console.log("First 5 values:", vector.slice(0, 5));
        } catch (error) {
            console.error("Embedding error:", error);
        }
    })();
}
