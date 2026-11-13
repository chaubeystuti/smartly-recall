
const { pipeline } = require("@huggingface/transformers");
const path = require("path");

let extractor;

async function getExtractor() {
    if (!extractor) {
        const modelPath = path.join(
            __dirname,
            "models",
            "fine_tuned_minilm"
        );

        console.log("Loading fine-tuned local embedding model...");

        extractor = await pipeline(
            "feature-extraction",
            modelPath,
            {
                local_files_only: true
            }
        );

        console.log("Fine-tuned model loaded!");
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
