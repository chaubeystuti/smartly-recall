
const faqs = require("./faqs");
const evaluationData = require("./eval");
const { getEmbedding } = require("./embedding");

function cosineSimilarity(a, b) {
    return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

async function evaluate() {
    console.log("\n========== SMARTLY RECALL EVALUATION ==========\n");

    // Step 1: Generate FAQ embeddings
    console.log("Generating FAQ embeddings...");

    const faqVectors = [];

    for (const faq of faqs) {
        const embedding = await getEmbedding(faq.question);

        faqVectors.push({
            id: faq.id,
            embedding
        });
    }

    console.log("FAQ embeddings generated!");

    // Step 2: Evaluate test questions
    console.log("\nEvaluating test questions...");

    const results = [];

    for (let i = 0; i < evaluationData.length; i++) {
        const item = evaluationData[i];
        const questionEmbedding = await getEmbedding(item.question);

        let bestMatch = null;
        let highestScore = -1;

        for (const faq of faqVectors) {
            const score = cosineSimilarity(
                questionEmbedding,
                faq.embedding
            );

            if (score > highestScore) {
                highestScore = score;
                bestMatch = faq;
            }
        }

        results.push({
            question: item.question,
            expectedFaqId: item.expectedFaqId,
            matchedFaqId: bestMatch.id,
            similarity: highestScore
        });

        console.log(`Evaluated ${i + 1}/${evaluationData.length}`);
    }

    // Step 3: Test thresholds
    const thresholds = [
        0.35, 0.40, 0.45, 0.50, 0.55,
        0.60, 0.65, 0.70, 0.75, 0.80, 0.85
    ];

    const totalPositive = evaluationData.filter(
        item => item.expectedFaqId !== null
    ).length;

    const totalNegative = evaluationData.filter(
        item => item.expectedFaqId === null
    ).length;

    console.log("\n========== THRESHOLD COMPARISON ==========\n");

    console.log(
        "Threshold | Correct | False | Misses | Precision | Recall | FPR"
    );

    for (const threshold of thresholds) {
        let correctHits = 0;
        let falseHits = 0;
        let misses = 0;

        let falsePositiveNegatives = 0;

        for (const result of results) {
            const isCacheHit = result.similarity >= threshold;

            if (result.expectedFaqId === null) {
                if (isCacheHit) {
                    falseHits++;
                    falsePositiveNegatives++;
                }
            } else if (!isCacheHit) {
                misses++;
            } else if (
                result.matchedFaqId === result.expectedFaqId
            ) {
                correctHits++;
            } else {
                falseHits++;
                misses++;
            }
        }

        // Precision = correct hits / all cache hits
        const precision = correctHits + falseHits > 0
            ? (correctHits / (correctHits + falseHits)) * 100
            : 0;

        // Recall = correct hits / all positive questions
        const recall = (correctHits / totalPositive) * 100;

        // False Positive Rate = negative questions incorrectly cached / all negative questions
        const fpr = (falsePositiveNegatives / totalNegative) * 100;

        console.log(
            `${threshold.toFixed(2)} | ${correctHits} | ${falseHits} | ${misses} | ${precision.toFixed(2)}% | ${recall.toFixed(2)}% | ${fpr.toFixed(2)}%`
        );

        // Show incorrect matches at threshold 0.65
        if (threshold === 0.65) {
            console.log(
                "\n========== FALSE HIT DETAILS (0.65) ==========\n"
            );

            for (const result of results) {
                if (
                    result.similarity >= threshold &&
                    (
                        result.expectedFaqId === null ||
                        result.matchedFaqId !== result.expectedFaqId
                    )
                ) {
                    console.log({
                        question: result.question,
                        expectedFaqId: result.expectedFaqId,
                        matchedFaqId: result.matchedFaqId,
                        similarity: Number(
                            result.similarity.toFixed(4)
                        )
                    });
                }
            }

            console.log(
                "\n==============================================\n"
            );
        }
    }

    console.log("\n========== EVALUATION COMPLETED ==========\n");
}

evaluate().catch(error => {
    console.error("Evaluation failed:", error);
});
