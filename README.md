# Handwritten Notes API

This is the single backend for the application. It combines the FastAPI model
inference API with the former HTTP gateway's Supabase authentication and
handwritten-notes processing route.

## Features

- Supabase signup, login, logout, and current-user endpoints
- Authenticated handwritten note processing with in-memory image uploads
- YOLO page detection and OpenCV preprocessing for image embeddings
- Jina Embeddings v4 image and question multivectors
- Configurable Gemini or OpenRouter vision answer generation
- Persistent Jina page embeddings in Qdrant Cloud
- Persistent processed page images in Supabase Storage

Document indexing stores page-level Jina multivectors in the
`handwritten_pages_jina` Qdrant collection and the corresponding processed PNGs
in the configured Supabase Storage bucket. The Qdrant payload contains the
storage key, so retrieval remains available across requests and API restarts.
Jina embeddings are generated through the cloud API. The selected answer
provider is called remotely, and no local Qwen3-VL weights are downloaded or
loaded. The YOLO page detector is loaded only for image embedding requests and
released afterward.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the CUDA-enabled build on an NVIDIA GPU, install the GPU requirements
instead:

```powershell
pip install -r requirements-gpu.txt
```

Copy `.env.example` to `.env` and set `FRONTEND_ORIGIN`, `SUPABASE_URL`,
`QDRANT_URL`, `QDRANT_API_KEY`, and `JINA_API_KEY`, plus the credentials for
the selected answer provider, and
`SUPABASE_PUBLISHABLE_KEY` (or `SUPABASE_ANON_KEY`), `SUPABASE_SERVICE_ROLE_KEY`,
and `SUPABASE_STORAGE_BUCKET`. Never use the Supabase service-role key as the
publishable key.

Place the YOLO weights at `models/best.pt` or set `YOLO_MODEL_PATH`.

Start the API:

```powershell
python -m app.main
```

`PORT` defaults to `3000`. Interactive API documentation is available at
`http://localhost:3000/docs`.

## Configuration

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `PORT` | No | `3000` | HTTP server port |
| `FRONTEND_ORIGIN` | Yes | — | Allowed browser origin for CORS |
| `SUPABASE_URL` | Yes | — | Supabase project URL |
| `SUPABASE_PUBLISHABLE_KEY` | Yes* | — | Supabase public API key |
| `SUPABASE_ANON_KEY` | Yes* | — | Legacy alternative to publishable key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes for document routes | — | Server-side Supabase Storage access |
| `SUPABASE_STORAGE_BUCKET` | No | `handwritten-files` | Supabase Storage bucket for page images |
| `MAX_IMAGE_COUNT` | No | `5` | Maximum images per handwritten request |
| `MAX_IMAGE_SIZE_BYTES` | No | `10485760` | Maximum bytes per image |
| `JINA_API_KEY` | Required for embeddings | — | Jina AI API key |
| `JINA_EMBEDDING_MODEL` | No | `jina-embeddings-v4` | Jina embedding model |
| `JINA_BASE_URL` | No | `https://api.jina.ai/v1` | Jina API base URL |
| `GEMINI_API_KEY` | Required for answers | — | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-3.8-flash` | Gemini answer-generation model |
| `LLM_PROVIDER` | No | `gemini` | Answer provider: `gemini` or `openrouter` |
| `OPENROUTER_API_KEY` | Required when selected | — | OpenRouter API key |
| `OPENROUTER_MODEL` | No | `qwen/qwen3-vl-8b-instruct` | OpenRouter vision model |
| `OPENROUTER_BASE_URL` | No | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `OPENROUTER_MAX_TOKENS` | No | `1024` | Maximum answer tokens |
| `OPENROUTER_TEMPERATURE` | No | `0.2` | Answer sampling temperature |
| `YOLO_MODEL_PATH` | No | `models/best.pt` | YOLO weights path |
| `QDRANT_URL` | Required for document routes | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Required for document routes | — | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | No | `handwritten_pages` | Persistent collection name |
| `JINA_QDRANT_COLLECTION` | No | `handwritten_pages_jina` | Jina vector collection name |

`SUPABASE_PUBLISHABLE_KEY` or `SUPABASE_ANON_KEY` must be set.

## API

Authentication routes preserve the former gateway's response envelopes:

- `POST /api/auth/signup` — create an account; returns `201 { "data": ... }`
- `POST /api/auth/login` — authenticate; returns `200 { "data": ... }`
- `GET /api/auth/me` — current user; requires a bearer token
- `POST /api/auth/logout` — revoke the session; requires a bearer token and returns `204`

Process one or more JPEG, PNG, or WebP pages:

```bash
curl -X POST http://localhost:3000/api/handwritten/process \
  -H "Authorization: Bearer <access-token>" \
  -F "question=Summarize these notes" \
  -F "images=@page-1.jpg" \
  -F "images=@page-2.png"
```

The response is `200 { "answer": "..." }`. Upload count and size are configured
above. Request errors use `{ "error": { "code": "...", "message": "..." } }`.

The inference routes remain available:

- `POST /api/answer` — multipart `question` and repeated `images`
- `POST /api/embeddings/image` — multipart `image`
- `POST /api/embeddings/question` — JSON `{"question":"..."}`
- `POST /api/documents/index` — multipart `document` and optional `notebook_id`
- `POST /api/documents/search` — JSON `{"query":"...", "top_k":5}`
- `POST /api/documents/ask` — JSON search request; answers from retrieved pages only
- `GET /api/documents` and `GET /api/documents/{document_id}` — indexed metadata
- `GET /api/documents/{document_id}/pages` — page metadata
- `GET /api/documents/{document_id}/pages/{page_id}/image` — stored page image
- `POST /api/documents/{document_id}/reindex` — re-embed legacy page images into the Jina collection
- `DELETE /api/documents/{document_id}` — delete embeddings and page images
- `GET /health` — health check

Example:

```bash
curl -X POST http://localhost:3000/api/documents/index \
  -F "document=@notebook-page.jpg" \
  -F "notebook_id=physics-notes"

curl -X POST http://localhost:3000/api/documents/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the formula for kinetic energy?","top_k":5}'
```

Jina Embeddings v4 returns a native matrix of visual/token vectors. The API
stores that matrix as a Qdrant multivector and configures Qdrant's MAX_SIM
comparator; it does not average or flatten the patches. The separate Jina
collection is created on first indexing and its dimensions/configuration are
validated before every write or query. Qdrant stores embeddings and metadata,
while PNG page files remain outside Qdrant.

## Model setup

Jina Embeddings v4 and Gemini Flash run remotely; no local embedding or
answer-generation transformer model is loaded. YOLO is loaded only during page
detection and released afterward.
Create the API key at
`https://aistudio.google.com/apikey` and keep it only in `.env`.

After installing the requirements, confirm the YOLO runtime can access the
intended PyTorch installation.
