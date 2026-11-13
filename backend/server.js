
const express = require("express");
const cors = require("cors");
require("dotenv").config();

const { initializeCache, searchCache } = require("./cache");

const app = express();
const exactCache = new Map();

let groqClient;

async function getGroqClient() {
    if (!groqClient) {
        const { default: Groq } = await import("groq-sdk");

        groqClient = new Groq({
            apiKey: process.env.GROQ_API_KEY
        });
    }

    return groqClient;
}

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
    res.send("Welcome to Smartly Recall API!");
});

app.get("/api/status", (req, res) => {
    res.json({
        project: "Smartly Recall",
        status: "Working"
    });
});

app.post("/api/ask", async (req, res) => {
    const { question } = req.body || {};

    if (typeof question !== "string" || !question.trim()) {
        return res.status(400).json({
            error: "Please provide a valid question"
        });
    }

    const normalizedQuestion = question.trim().toLowerCase();

    // STEP 1: Exact cache
    if (exactCache.has(normalizedQuestion)) {
        return res.json({
            answer: exactCache.get(normalizedQuestion),
            cacheHit: true,
            cacheType: "exact"
        });
    }

    try {
        // STEP 2: Semantic cache
        const semanticResult = await searchCache(question);

        if (semanticResult.cacheHit) {
            return res.json(semanticResult);
        }

        // STEP 3: Call Groq if no cache match
        const groq = await getGroqClient();

        const response = await groq.chat.completions.create({
            messages: [
                {
                    role: "system",
                    content: "You are a helpful e-commerce customer support assistant. Answer clearly and concisely."
                },
                {
                    role: "user",
                    content: question
                }
            ],
            model: "openai/gpt-oss-120b"
        });

        const answer = response.choices[0].message.content;

        exactCache.set(normalizedQuestion, answer);

        return res.json({
            answer,
            cacheHit: false,
            cacheType: "miss",
            similarity: semanticResult.similarity
        });

    } catch (error) {
        console.error("API Error:", error.message);

        return res.status(502).json({
            error: "Unable to generate an answer right now. Please try again."
        });
    }
});

// Load FAQ embeddings before accepting requests
async function startServer() {
    try {
        await initializeCache();

        const PORT = 5000;

        app.listen(PORT, () => {
            console.log(`Smartly Recall running on port ${PORT}`);
        });

    } catch (error) {
        console.error("Failed to initialize semantic cache:", error);
        process.exit(1);
    }
}

startServer();
