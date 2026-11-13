
const { pipeline } = require("@huggingface/transformers");
const path = require("path");

async function testModel() {
    try {
        const modelPath = path.join(
            __dirname,
            "models",
            "fine_tuned_minilm"
        );

        console.log("Loading fine-tuned local model...");

        const extractor = await pipeline(
            "feature-extraction",
            modelPath,
            {
                local_files_only: true
            }
        );

        console.log("Model loaded successfully!");

        const output = await extractor(
            "How can I track my order?",
            {
                pooling: "mean",
                normalize: true
            }
        );

        const vector = Array.from(output.data);

        console.log("Embedding generated successfully!");
        console.log("Vector dimensions:", vector.length);
        console.log("First 5 values:", vector.slice(0, 5));

    } catch (error) {
        console.error("MODEL TEST FAILED:");
        console.error(error);
    }
}

testModel();
