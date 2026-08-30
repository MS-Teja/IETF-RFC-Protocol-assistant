# Protocol Assistant: IETF RFC Knowledge Base

## Project Name
**Protocol Assistant**

## Problem Statement
Developers frequently need to consult complex, dense technical documentation and standard specifications (IETF RFCs) when building networking applications, web servers, or API clients. Navigating hundreds of pages of RFCs to find specific implementation details (e.g., HTTP caching semantics, JWT validation rules, WebSocket framing) is time-consuming and prone to human error. General-purpose LLMs often hallucinate or blend different versions of protocols when asked these questions.

## Chatbot Use Case
Protocol Assistant is a specialized, AI-powered Retrieval-Augmented Generation (RAG) chatbot designed to act as an expert Protocol Engineer. It strictly grounds its answers in 9 canonical IETF RFCs covering the core protocols of the web: HTTP/1.1-HTTP/3, TLS 1.3, OAuth 2.0, JWT, and WebSockets. It is built to provide precise, citeable answers to highly technical protocol questions, making it an invaluable tool for software engineers.

## Key Features
- **Strict Grounding:** Answers are generated *only* from the provided RFCs, virtually eliminating hallucinations on technical specs.
- **Precise Citations:** Every answer includes explicit citations back to the source RFC and specific section (e.g., `[RFC 9110, §15.5.1]`).
- **Conversational Memory:** Maintains recent chat history context to answer follow-up questions accurately.
- **Multi-Model Support with Fallback:** Uses Claude (Anthropic) as the primary intelligence, with automatic seamless failover to Gemini (Google) if Claude is rate-limited or unavailable.
- **Beautiful UI:** A completely custom, responsive, Apple-inspired interface built with raw HTML/CSS/JS (no heavy frontend frameworks), featuring markdown rendering, code syntax highlighting, dark mode, and seamless session history via `localStorage`.
- **Graceful Degradation:** Real-time health API monitoring and robust error handling that allows the user to retry failed requests with a different model without losing conversation context.

## Technology Stack
- **Backend Framework:** FastAPI (Python 3.12+)
- **Frontend:** Vanilla HTML, CSS, JavaScript
- **Vector Database:** ChromaDB (Local, Persistent)
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`) via HuggingFace
- **LLM Integrations:** `anthropic` and `google-genai` official Python SDKs
- **Containerization:** Docker

## Selected AI/ML Models
- **Primary LLM:** Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`) or Claude 3.5 Haiku (`claude-3-5-haiku-20241022`) via Anthropic. Chosen for their superior technical reasoning and coding capabilities.
- **Fallback LLM:** Gemini 1.5 Flash (`gemini-1.5-flash`) via Google. Chosen for its speed and massive context window as a highly reliable secondary provider.
- **Embedding Model:** `all-MiniLM-L6-v2`. Chosen for its excellent balance of speed and semantic representation in a lightweight footprint (384 dimensions) that runs easily on CPU without external API calls.

## RAG Architecture
The system uses a classic Retrieval-Augmented Generation pipeline:
1. **Contextualization:** The user's query is combined with recent conversation history to generate a standalone contextualized query.
2. **Retrieval:** The query is embedded, and the vector store is queried using Cosine Similarity.
3. **Filtering:** Results are filtered using a hard similarity threshold (0.3) to reject irrelevant queries.
4. **Prompt Injection:** The retrieved chunks are formatted with explicit `[Source N]` labels and injected into a strict system prompt.
5. **Generation:** The LLM generates the final answer using *only* the injected context, forcing citations to the provided labels.

## Data Sources
The corpus consists of 9 canonical IETF RFCs sourced directly from the IETF Datatracker:
- **HTTP:** RFC 9110 (Semantics), 9111 (Caching), 9112 (HTTP/1.1), 9113 (HTTP/2), 9114 (HTTP/3)
- **Security:** RFC 8446 (TLS 1.3), RFC 6749 (OAuth 2.0), RFC 7519 (JWT)
- **Transport:** RFC 6455 (WebSockets)

## Embedding Approach
Documents are pre-processed using a custom Regex-based chunker that splits the text intelligently at RFC section boundaries (e.g., `1. Introduction`, `4.1.2. Constraints`). This ensures that chunks remain semantically coherent and contain complete thoughts. The chunks are then embedded locally using the `SentenceTransformerEmbedder` which runs in-process to avoid network latency and API costs for embeddings.

## Vector Database/Retrieval Approach
**ChromaDB** is used as the vector store, configured in local persistent mode (`PersistentClient`). It stores both the vectors and the rich metadata (RFC document name, section ID, and content hash). 
During retrieval, a **Cosine Similarity** metric is used (Chroma default is L2, but we extract the scores and normalize them). We use a fixed `similarity_threshold` (e.g., 0.3) to aggressively filter out low-confidence matches. If a user asks a question unrelated to the RFCs (e.g., "What is the capital of France?"), the retrieval step returns 0 chunks, and the LLM safely declines to answer.

## Project Architecture
The backend follows a strict layered architecture to ensure separation of concerns and maintainability:
- **`app/api/`**: FastAPI route handlers (HTTP interface).
- **`app/rag/`**: Core RAG business logic (`pipeline.py`, `ingestion.py`, `prompts.py`).
- **`app/services/`**: Abstract interfaces and concrete implementations for external dependencies (LLMs, Vector DB, Embedder).
- **`app/models/`**: Pydantic schemas for request/response validation.
- **`app/core/`**: Configuration and Dependency Injection wiring.

## Code Modularity & Extensibility
The application is strictly interface-driven, making it trivial to extend or swap components without rewriting core logic:
- **Replacing the LLM:** `app/services/llm_service.py` defines an `LLMService` abstract base class. To add OpenAI, simply create an `OpenAILLM(LLMService)` class and swap it in the dependency injection container (`app/core/dependencies.py`).
- **Replacing Embeddings:** Similarly, `app/services/embedding_service.py` defines an `EmbeddingService` interface. The default uses `SentenceTransformers`, but can be swapped for an API-based embedder without modifying the RAG pipeline.
- **Adding Data Sources:** `app/rag/ingestion.py` uses modular loaders (like `RFCLoader`). New loaders for PDFs, Web scraping, or databases can be added by implementing a similar `load()` method and appending the chunks to the ingestion pipeline.

## Directory Structure
```
cyberlex-india/
├── app/
│   ├── api/            # API endpoints (chat, health)
│   ├── core/           # Config and DI dependencies
│   ├── models/         # Pydantic schemas
│   ├── rag/            # RAG pipeline, ingestion, prompts
│   ├── services/       # Service integrations (LLMs, Chroma)
│   └── main.py         # FastAPI application entry point
├── data/
│   └── rfcs/           # Raw text RFC documents
├── static/
│   └── index.html      # Frontend UI
├── chroma_data/        # Persistent vector store (generated)
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variable template
```

## API Information
- `GET /`: Serves the static HTML frontend.
- `GET /health`: Returns system health, connected LLM provider, and number of indexed vector chunks.
- `POST /chat`: The main conversational endpoint. Accepts a JSON payload with `query`, `history`, and `model`. Returns the `answer`, `citations`, and `model_used`.

## Local Setup Instructions

1. **Clone the repository** and navigate to the project root.
2. **Set up a Python virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables:**
   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env and set ANTHROPIC_API_KEY and/or GOOGLE_API_KEY
   ```
5. **Run the initial data ingestion** (This parses the RFCs and builds the Chroma database):
   ```bash
   python test_ingest.py
   ```
6. **Start the API server:**
   ```bash
   uvicorn app.main:app --reload
   ```
7. **Access the application:** Open `http://localhost:8000` in your browser.

## Environment Variables
The application requires API keys to function. It uses a `.env` file for local development.

- `LLM_PROVIDER`: The primary LLM to use (`claude` or `gemini`).
- `ANTHROPIC_API_KEY`: Required if using Claude.
- `GOOGLE_API_KEY`: Required if using Gemini.
- `CHROMA_PATH`: Path to store the vector database (default: `./chroma_data`).
- `EMBEDDING_MODEL`: The HuggingFace model string for embeddings (default: `all-MiniLM-L6-v2`).
- `HF_HUB_OFFLINE`: Set to `1` in Docker to prevent HuggingFace from attempting network calls.

## Docker Build Instructions
To build the Docker image locally:
```bash
docker build -t protocol-assistant:latest .
```

## Docker Run Instructions
To run the Docker container, you should pass your configuration via a `.env` file and mount a volume for the vector database so you don't have to re-ingest data every time the container restarts.

Ensure you have created a `.env` file from the example:
```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY and/or GOOGLE_API_KEY
```

```bash
# Run the container
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/chroma_data:/app/chroma_data \
  --name protocol-assistant \
  protocol-assistant:latest
```
*Note: The container runs as a non-root user (`appuser`) for security.*

## Docker Hub Image Information
The pre-built Docker image is available on Docker Hub and is **100% self-contained** (the embedding model and all 1,101 RFC chunks are pre-baked into the image).

- **Repository:** `m5teja/protocol-assistant`
- **Tag:** `latest`

### Option A: Standalone Run (No Git Clone Required)
Evaluators can pull and run the container immediately with their API keys:
```bash
docker pull m5teja/protocol-assistant:latest

# Recommended: Run with both keys (enables model switching and automatic fallback)
docker run -d -p 8000:8000 \
  -e ANTHROPIC_API_KEY="your-anthropic-api-key" \
  -e GOOGLE_API_KEY="your-google-api-key" \
  --name protocol-assistant \
  m5teja/protocol-assistant:latest

# Or run with a single key (e.g. Gemini only):
docker run -d -p 8000:8000 -e GOOGLE_API_KEY="your-google-api-key" --name protocol-assistant m5teja/protocol-assistant:latest
```

### Option B: Run with Cloned Repository (.env File)
If the repository is cloned locally:
```bash
docker pull m5teja/protocol-assistant:latest

docker run -d \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/chroma_data:/app/chroma_data \
  --name protocol-assistant \
  m5teja/protocol-assistant:latest
```

## Known Limitations
- **Ingestion Time:** The initial `test_ingest.py` script takes a few minutes to run because it embeds thousands of chunks locally on the CPU.
- **Context Window:** The conversation history is artificially limited to the last 4 messages to prevent the prompt size from exceeding LLM limits and to keep costs down.
- **Query Contextualization:** Currently, query contextualization uses a basic heuristic rather than a dedicated LLM call, which can sometimes fail on highly complex, multi-turn implicit pronoun references.

## Important Implementation Decisions
1. **Dependency Injection:** The entire application uses FastAPI's `Depends` system for service instantiation (LLMs, Embedders, Vector DB). This allows for extremely easy unit testing and mocking. 
2. **Local Embeddings:** Chose to run `SentenceTransformers` locally rather than using OpenAI/Cohere embeddings. This significantly reduces operating costs and network latency.
3. **No Frontend Framework:** Built the frontend using raw HTML/CSS/JS in a single file to keep the project lightweight and remove the need for a separate Node.js build pipeline, while still achieving a premium, native-feeling UI.
4. **Non-Root Docker Container:** The Dockerfile is specifically configured to create and run as a non-root `appuser` to adhere to modern container security best practices.
5. **Robust Error Handling:** API errors from upstream LLM providers (like HTTP 503s from Gemini) are propagated securely and caught by the API layer to return proper HTTP 502 Bad Gateway responses, rather than silently failing or returning 200 OKs with error text.
