
const { getEmbedding } = require("./embedding");

// Calculate cosine similarity using normalized vectors
function similarity(a, b) {
    return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

async function test() {
    const pairs = [
        // POSITIVE MATCHES: Same meaning, different wording
        {
            type: "POSITIVE",
            question1: "How can I track my order?",
            question2: "How do I track my package?"
        },
        {
            type: "POSITIVE",
            question1: "How can I track my order?",
            question2: "Where is my order?"
        },
        {
            type: "POSITIVE",
            question1: "How can I track my order?",
            question2: "Can I check my shipment status?"
        },

        // NEGATIVE MATCHES: Different meanings
        {
            type: "NEGATIVE",
            question1: "How can I track my order?",
            question2: "How can I change my shipping address?"
        },
        {
            type: "NEGATIVE",
            question1: "How can I track my order?",
            question2: "What payment methods are accepted?"
        },
        {
            type: "NEGATIVE",
            question1: "How can I track my order?",
            question2: "How do I cancel my order?"
        }
    ];

    console.log("\n========== SEMANTIC SIMILARITY TEST ==========\n");

    for (const pair of pairs) {
        const embedding1 = await getEmbedding(pair.question1);
        const embedding2 = await getEmbedding(pair.question2);

        const score = similarity(embedding1, embedding2);

        console.log("----------------------------------------");
        console.log(`Type: ${pair.type}`);
        console.log(`Question 1: ${pair.question1}`);
        console.log(`Question 2: ${pair.question2}`);
        console.log(`Similarity: ${score.toFixed(4)}`);
    }

    console.log("\n========== TEST COMPLETED ==========\n");
}

test().catch(error => {
    console.error("Test failed:", error);
});
