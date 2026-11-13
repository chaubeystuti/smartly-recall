
const faqs = require("./faqs");
const { getEmbedding } = require("./embedding");

// Our in-memory vector cache
const vectorCache = [];

// Similarity threshold
const SIMILARITY_THRESHOLD = 0.65;

// Load all FAQs into the cache
async function initializeCache() {
    console.log("Initializing semantic cache...");

    for (const faq of faqs) {
        const embedding = await getEmbedding(faq.question);

        vectorCache.push({
            id: faq.id,
            question: faq.question,
            answer: faq.answer,
            embedding
        });

        console.log(`Cached FAQ ${faq.id}/30`);
    }

    console.log("Semantic cache initialized!");
}

// Calculate cosine similarity
function cosineSimilarity(vectorA, vectorB) {
    let dotProduct = 0;

    for (let i = 0; i < vectorA.length; i++) {
        dotProduct += vectorA[i] * vectorB[i];
    }

    return dotProduct;
}

// Find the closest matching FAQ
async function searchCache(userQuestion) {
    const userEmbedding = await getEmbedding(userQuestion);

    let bestMatch = null;
    let highestScore = -1;

    for (const item of vectorCache) {
        const score = cosineSimilarity(
            userEmbedding,
            item.embedding
        );

        if (score > highestScore) {
            highestScore = score;
            bestMatch = item;
        }
    }

    if (bestMatch && highestScore >= SIMILARITY_THRESHOLD) {
        return {
            cacheHit: true,
            cacheType: "semantic",
            answer: bestMatch.answer,
            matchedQuestion: bestMatch.question,
            similarity: Number(highestScore.toFixed(4))
        };
    }

    return {
        cacheHit: false,
        cacheType: "miss",
        similarity: Number(highestScore.toFixed(4))
    };
}

module.exports = {
    initializeCache,
    searchCache
};
