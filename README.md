
# Smartly Recall
### AI-Powered Semantic Caching System

Smartly Recall is an AI-powered FAQ chatbot backend that uses semantic caching to retrieve relevant answers based on the meaning of user queries rather than relying only on exact text matches.

It combines exact caching, semantic similarity search, locally deployed fine-tuned multilingual embeddings, and Groq LLM fallback to improve response speed and reduce unnecessary LLM API calls.

## Key Features

- **Exact Caching:** Stores and retrieves answers for previously asked identical questions.
- **Semantic Caching:** Understands questions with similar meanings, even when their wording differs.
- **Multilingual Embeddings:** Uses a fine-tuned multilingual MiniLM model.
- **Local Model Inference:** Runs the embedding model locally using ONNX.
- **LLM Fallback:** Uses Groq when no relevant cached FAQ is found.
- **Similarity Threshold:** Controls when a semantic match is accepted.
- **Performance Evaluation:** Evaluates semantic matching using precision, recall and false positive rate.
- **API Benchmarking:** Measures semantic cache, exact cache and LLM fallback latency.

## System Architecture

```text
             User Question
                   |
                   v
             Node.js API
                   |
                   v
              Exact Cache
                   |
          +--------+--------+
          |                 |
        HIT                MISS
          |                 |
          v                 v
      Return Answer    Semantic Cache
                            |
                            v
                   Fine-Tuned MiniLM
                      (ONNX Model)
                            |
                            v
                   Vector Similarity
                      Calculation
                            |
                 +----------+----------+
                 |                     |
               HIT                    MISS
                 |                     |
                 v                     v
           Return FAQ Answer       Groq LLM
                                       |
                                       v
                                  Return Answer
                                       |
                                       v
                               Store in Exact Cache
```

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Node.js |
| API Framework | Express.js |
| Embedding Model | Multilingual MiniLM |
| Model Training | Python, PyTorch |
| Fine-Tuning | Sentence Transformers |
| Model Deployment | ONNX |
| Model Inference | Hugging Face Transformers.js |
| LLM Fallback | Groq API |
| Similarity Metric | Cosine Similarity |
| Evaluation | Precision, Recall, FPR |

## Project Structure

```text
smartlyrecall/
│
├── backend/
│   ├── models/
│   │   └── fine_tuned_minilm/
│   ├── server.js
│   ├── cache.js
│   ├── embedding.js
│   ├── faqs.js
│   ├── evaluate.js
│   ├── benchmark.js
│   ├── api-benchmark.js
│   └── package.json
│
├── training/
│   ├── train_model.py
│   ├── training_pairs.csv
│   ├── evaluation_data.json
│   └── fine_tuned_model/
│
└── README.md
```

## How It Works

1. A user sends a question to the backend API.
2. The system checks whether an identical question exists in the exact cache.
3. If not, the question is converted into a 384-dimensional embedding using the locally deployed fine-tuned MiniLM model.
4. Cosine similarity is calculated against stored FAQ embeddings.
5. If the highest similarity score meets the configured threshold of 0.65, the corresponding FAQ answer is returned.
6. If no suitable semantic match is found, the question is sent to Groq.
7. The generated answer is stored in the exact cache for future identical queries.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd smartlyrecall
```

### 2. Install backend dependencies

```bash
cd backend
npm install
```

### 3. Configure environment variables

Create a `.env` file inside the backend folder:

```env
GROQ_API_KEY=your_groq_api_key
```

### 4. Start the backend

```bash
node server.js
```

The server runs on:

```text
http://localhost:5000
```

## API Usage

### Endpoint

```http
POST /api/ask
```

### Request

```json
{
  "question": "Where can I track my order?"
}
```

### Example Response

```json
{
  "answer": "You can track your order through the My Orders section of your account.",
  "cacheHit": true,
  "cacheType": "semantic",
  "matchedQuestion": "How can I track my order?",
  "similarity": 0.89
}
```

*Example response structure; actual similarity values depend on the query.*

## Model Training

The semantic embedding model was fine-tuned using a custom dataset of 192 question pairs:

- 60 positive pairs
- 132 negative pairs
- Hard-negative examples to improve distinction between similar FAQ topics.

The model was fine-tuned using Sentence Transformers and exported to ONNX for local inference.

Embedding dimension: **384**

## Evaluation Results

The fine-tuned model was evaluated using 100 test questions.

At a similarity threshold of 0.65:

| Metric | Result |
|---|---:|
| Correct semantic matches | 42 |
| False semantic matches | 1 |
| Misses | 18 |
| Precision | 97.67% |
| Recall | 70.00% |
| False Positive Rate | 0.00% |

These results are based on the current evaluation dataset and are not a guarantee of performance on all user queries.

## Performance Benchmark

Measured through the local API benchmark:

| Operation | Average Latency |
|---|---:|
| Semantic Cache Hit | 13.35 ms |
| Exact Cache Hit | 2.91 ms |
| Groq Fallback | 2093.00 ms |

Benchmark details:
- Semantic cache: 10 requests.
- Exact cache: 10 requests.
- Groq fallback: 1 request.

These are observed results from one benchmark run on the development machine.

## Future Improvements

- Persistent cache storage using Redis or a database.
- Larger and more diverse FAQ datasets.
- Improved semantic matching and hard-negative training.
- Production deployment and monitoring.
- Authentication and rate limiting.

## Author

Developed as an AI/ML backend project exploring semantic search, embedding fine-tuning, model deployment and intelligent caching.
