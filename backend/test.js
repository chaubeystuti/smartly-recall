
require("dotenv").config();

async function testGroq() {
    try {
        const { default: Groq } = await import("groq-sdk");

        const groq = new Groq({
            apiKey: process.env.GROQ_API_KEY
        });

        const response = await groq.chat.completions.create({
            messages: [
                {
                    role: "user",
                    content: "Explain machine learning in 3 simple sentences."
                }
            ],
            model: "openai/gpt-oss-120b"
        });

        console.log("Groq Response:");
        console.log(response.choices[0].message.content);

    } catch (error) {
        console.error("Groq connection failed:", error.message);
    }
}

testGroq();
