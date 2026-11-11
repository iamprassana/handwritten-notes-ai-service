# Handwritten Notes API

This is the single backend for the application. It combines the FastAPI model
inference API with the former HTTP gateway's Supabase authentication and
handwritten-notes processing route.

## Features

- Supabase signup, login, logout, and current-user endpoints
- Authenticated handwritten note processing with in-memory image uploads
- YOLO page detection and OpenCV preprocessing for image embeddings
- ColQwen2 image and question embeddings
- Qwen3-VL grounded answer generation
- A reusable Supabase Storage service for future storage routes

The API does not currently expose database, Qdrant, or storage routes. Images
uploaded for note processing are held in memory and are not persisted.
Inference models load only when first needed. ColQwen2 and Qwen3-VL are swapped
one at a time, and the YOLO page detector is loaded only for image-embedding
requests and released afterward. This reduces idle and peak memory use; the
first request for a model takes longer while it loads.

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

Copy `.env.example` to `.env` and set `FRONTEND_ORIGIN`, `SUPABASE_URL`, and
`SUPABASE_PUBLISHABLE_KEY` (or `SUPABASE_ANON_KEY`). Never use the Supabase
service-role key as the publishable key. The service-role key is optional and
only needed by the unused storage service.

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
| `SUPABASE_SERVICE_ROLE_KEY` | No | — | Only used by the storage service |
| `SUPABASE_STORAGE_BUCKET` | No | `handwritten-files` | Storage service bucket |
| `MAX_IMAGE_COUNT` | No | `5` | Maximum images per handwritten request |
| `MAX_IMAGE_SIZE_BYTES` | No | `10485760` | Maximum bytes per image |
| `COLQWEN_MODEL` | No | `vidore/colqwen2-v1.0-hf` | ColQwen2 model |
| `QWEN_MODEL` | No | `Qwen/Qwen3-VL-2B-Instruct` | Qwen3-VL model |
| `YOLO_MODEL_PATH` | No | `models/best.pt` | YOLO weights path |

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
- `GET /health` — health check

## Model setup

ColQwen2 and Qwen3-VL load on first use and are not kept in memory together.
On GPUs, Accelerate's automatic device mapping can split a model between GPU
and CPU when GPU memory is limited. YOLO runs on CPU. The model implementations
and dependency versions should match the validated notebook. The transformer
models may download from Hugging Face on their first request; configure any
required Hugging Face access before deploying.

After installing the GPU requirements, confirm PyTorch detects the GPU with
`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

The models still need enough system RAM individually, plus room for inference
activations. If the machine cannot fit either model by itself, unloading the
other model will not be sufficient; use a machine with more available memory
or a smaller model checkpoint.
